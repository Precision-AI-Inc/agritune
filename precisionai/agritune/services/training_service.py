# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Training orchestration — wires dataset + feature store + task + trainer into ``agritune train``.

Supports every ``augmentation.mode`` x ``feature_provider`` combination the plan's config
hierarchy (§23) calls for, except ``cached`` combined with ``online``/``hybrid`` augmentation —
rejected up front, since a cached provider's key would only ever match the first epoch's
augmentation fingerprint. This is also the module the plan's central integration test exercises:
tiny dataset -> fake encoder -> feature precompute -> decoder -> train -> checkpoint -> resume ->
evaluate.
"""

import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

import torch

from precisionai.agritune.augmentations.feature.pipeline import FeatureAugmentationConfig, FeatureAugmentationPipeline
from precisionai.agritune.augmentations.image.pipeline import (
    AugmentationMode,
    AugmentationPipelineConfig,
    GeometricConfig,
    ImageAugmentationPipeline,
    PhotometricConfig,
    prepare_sample,
)
from precisionai.agritune.data.dataset import ManifestDataset
from precisionai.agritune.data.manifest import ManifestRow, load_manifest
from precisionai.agritune.data.split import SplitAssignment, random_split
from precisionai.agritune.features.keys import EncoderFingerprint
from precisionai.agritune.features.provider import HybridFeatureProvider, OnlineFeatureProvider
from precisionai.agritune.features.store import FEATURE_STORE_TYPES, DirectoryFeatureStore, ShardedFeatureStore
from precisionai.agritune.logging import RunDirectory, get_logger, progress_iter
from precisionai.agritune.optimization.optimizers import OptimizerConfig, build_optimizer
from precisionai.agritune.optimization.schedulers import SchedulerConfig, build_scheduler
from precisionai.agritune.schemas.protocols import FeatureProvider, Tracker
from precisionai.agritune.schemas.samples import PreparedSample
from precisionai.agritune.services.encoder_selection import build_encoder
from precisionai.agritune.services.segmentation_common import (
    OnlineAugmentedBatches,
    build_cached_feature_provider,
    build_decoder,
    build_image_hash_fn,
    build_static_augmented_batches,
    build_training_batches,
    mask_output_size,
    probe_feature_dims,
    validate_device,
)
from precisionai.agritune.services.tracking_selection import TrackingSelection, build_trackers
from precisionai.agritune.tasks.segmentation.losses import SegmentationLoss, SegmentationLossConfig
from precisionai.agritune.tasks.segmentation.metrics import SegmentationMetric
from precisionai.agritune.tasks.segmentation.task import SegmentationTask
from precisionai.agritune.training.checkpointing import CheckpointManager
from precisionai.agritune.training.trainer import Trainer, TrainerConfig

_FEATURE_PROVIDERS = ("cached", "online", "hybrid")
logger = get_logger(__name__)


def _jsonable(value: Any) -> Any:
    result = value
    if isinstance(value, torch.Tensor):
        result = value.detach().cpu().tolist()
    elif isinstance(value, Enum):
        result = value.value
    elif isinstance(value, Path):
        result = str(value)
    elif isinstance(value, dict):
        result = {str(key): _jsonable(item) for key, item in value.items()}
    elif isinstance(value, list | tuple):
        result = [_jsonable(item) for item in value]
    elif is_dataclass(value) and not isinstance(value, type):
        result = _jsonable(asdict(value))
    return result


def _redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized.endswith(("api_key", "api_token", "authorization", "password")):
                redacted[str(key)] = "<redacted>" if item is not None else None
            else:
                redacted[str(key)] = _redact_secrets(item)
        return redacted
    if isinstance(value, list | tuple):
        return [_redact_secrets(item) for item in value]
    return _jsonable(value)


def _stable_fingerprint(value: Any) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dataset_fingerprint(manifest_path: str, rows: list[ManifestRow], *, show_progress: bool = False) -> str:
    """Hash the manifest plus every row's image/mask bytes, in ``sample_id`` order.

    Reads every file in ``rows`` off disk synchronously — for a large manifest this can take
    a while, so progress is logged up front and (optionally) rendered as a bar rather than
    running silently.
    """
    manifest = Path(manifest_path)
    sorted_rows = sorted(rows, key=lambda item: item.sample_id)
    logger.info("computing dataset fingerprint over %d sample(s); this reads every file once", len(sorted_rows))
    digest = hashlib.sha256(manifest.read_bytes())
    for row in progress_iter(sorted_rows, desc="fingerprinting dataset", unit="sample", disable=not show_progress):
        digest.update(row.sample_id.encode("utf-8"))
        digest.update((manifest.parent / row.image_path).read_bytes())
        digest.update((manifest.parent / row.mask_path).read_bytes())
    return digest.hexdigest()


def _critical_config(config: "TrainingRunConfig") -> dict[str, Any]:
    trainer = _jsonable(config.trainer)
    trainer.pop("max_epochs", None)
    trainer.pop("fingerprints", None)
    return {
        "num_classes": config.num_classes,
        "feature_provider": config.feature_provider,
        "encoder_base_url": config.encoder_base_url,
        "augmentation": _jsonable(config.augmentation),
        "feature_augmentation": _jsonable(config.feature_augmentation),
        "decoder_name": config.decoder_name,
        "decoder_kwargs": _jsonable(config.decoder_kwargs),
        "batch_size": config.batch_size,
        "val_fraction": config.val_fraction,
        "seed": config.seed,
        "optimizer": _jsonable(config.optimizer),
        "scheduler": _jsonable(config.scheduler),
        "trainer": trainer,
        "loss": _jsonable(config.loss),
        "val_metric_name": config.val_metric_name,
        "higher_is_better": config.higher_is_better,
    }


def _resolved_config(config: "TrainingRunConfig") -> dict[str, Any]:
    resolved: dict[str, Any] = _jsonable(config)
    resolved.pop("original_config", None)
    resolved.pop("config_overrides", None)
    return _redact_secrets(resolved)


def _original_config(config: "TrainingRunConfig") -> dict[str, Any]:
    source = config.original_config or _resolved_config(config)
    original = _redact_secrets(source)
    if config.config_overrides:
        overrides = []
        for override in config.config_overrides:
            key, separator, _ = override.partition("=")
            normalized = key.lower()
            if separator and normalized.endswith(("api_key", "api_token", "authorization", "password")):
                overrides.append(f"{key}=<redacted>")
            else:
                overrides.append(override)
        return {"config": original, "overrides": overrides}
    return original


@dataclass
class AugmentationSelection:
    """Which image-augmentation mode to apply to training samples, and its transform config.

    Attributes
    ----------
    mode : AugmentationMode
        ``NONE`` (default — no real augmentation: no flips/crops/photometric transforms), ``OFFLINE``,
        ``ONLINE``, or ``HYBRID``. ``geometric.resize`` is the one exception: applied to every
        training *and* validation batch regardless of ``mode`` — deterministic dimensional
        normalization, not augmentation, required whenever the dataset's own images/masks do not
        already share one native size (``torch.stack`` cannot batch unequal-size targets together).
    variant : int
        Offline variant index; ignored outside ``OFFLINE``/``HYBRID``.
    hybrid_online_probability : float
        Forwarded to ``prepare_sample`` under ``HYBRID``; ignored otherwise.
    geometric : GeometricConfig
    photometric : PhotometricConfig
    """

    mode: AugmentationMode = AugmentationMode.NONE
    variant: int = 0
    hybrid_online_probability: float = 0.3
    geometric: GeometricConfig = field(default_factory=GeometricConfig)
    photometric: PhotometricConfig = field(default_factory=PhotometricConfig)


@dataclass
class TrainingRunConfig:
    """Fully resolved configuration for one :func:`run_training`.

    Attributes
    ----------
    manifest_path : str
        Dataset manifest CSV.
    feature_store_dir : str
        Directory a feature store was already built into (via ``agritune features build`` /
        :mod:`feature_service`) — read from under ``feature_provider: cached``/``hybrid``, written
        to (write-through) under ``hybrid``, unused under ``online``.
    store_type : str
        ``"directory"`` (:class:`~precisionai.agritune.features.store.DirectoryFeatureStore`,
        development scale) or ``"sharded"``
        (:class:`~precisionai.agritune.features.store.ShardedFeatureStore`, production scale) —
        must match whatever ``feature_store_dir`` was actually built as.
    entries_per_shard : int
        Samples packed per shard file; only consulted when ``store_type == "sharded"``.
    run_root : str
        Directory ``runs/<run_id>/`` is created under.
    run_id : str
        Unique run identifier.
    num_classes : int
        Number of segmentation classes.
    encoder_fingerprint : EncoderFingerprint
        Must match what the feature store was built with under ``cached``/``hybrid``, or every
        lookup misses — see :class:`~precisionai.agritune.features.provider.CachedFeatureProvider`.
    feature_provider : str
        ``"cached"``, ``"online"``, or ``"hybrid"`` — see
        :mod:`precisionai.agritune.features.provider`.
    encoder_base_url : str | None
        Hosted encoder API base URL; ``None`` uses ``FakeEncoderBackend``. Only consulted when
        ``feature_provider`` is ``"online"``/``"hybrid"``.
    encoder_api_key : str | None
        Ignored when ``encoder_base_url`` is ``None``.
    augmentation : AugmentationSelection
    feature_augmentation : FeatureAugmentationConfig
        Applied to training batches' features only, right after ``feature_provider.get_features``
        and before the decoder — composes with ``augmentation`` (image-space) and any
        ``feature_provider`` freely, since it never touches the feature cache/store, just a
        runtime perturbation. Never applied to validation.
    decoder_name : str
        ``"mlp_probe"``, ``"token_fpn"``, ``"aspp"``, ``"ppm"``, ``"segmenter"``, or
        ``"mask_former"`` — see :func:`~precisionai.agritune.services.segmentation_common.build_decoder`.
    decoder_kwargs : dict[str, Any]
        Extra keyword arguments forwarded to the decoder's constructor (e.g. ``hidden_dims`` for
        ``"mlp_probe"``, ``atrous_rates`` for ``"aspp"``).
    batch_size : int
    device : str
        Where the decoder and every batch's features/targets are moved before ``forward``/
        ``compute_loss`` — ``"cpu"`` (the default, always safe on CPU-only machines), ``"cuda"``,
        or a specific GPU like ``"cuda:3"``. Rejected up front if it names a CUDA device that
        either isn't available at all or is out of range for this machine's GPU count — this never
        silently falls back to CPU.
    num_workers : int
        Forwarded to the ``DataLoader`` backing every train/validation batches iterable — worker
        subprocesses to decode/augment samples in parallel. ``0`` runs everything in the main
        process; purely a performance knob, so it is excluded from ``_critical_config`` and never
        affects a run's reproducibility.
    pin_memory : bool
        Forwarded to the same ``DataLoader``. Speeds up the host-to-device copy of batch targets
        when training on a CUDA device; has no effect on CPU-only runs.
    prefetch_factor : int | None
        Forwarded to the same ``DataLoader`` — batches each worker buffers ahead of time; ``None``
        defers to ``DataLoader``'s own default (``2``) and requires ``num_workers > 0``. Peak
        memory scales with ``num_workers * prefetch_factor * batch_size``, so a large ``batch_size``
        combined with many workers at the default of ``2`` can buffer enough whole batches at once
        to exhaust memory — cap this explicitly (e.g. ``1``) in that case.
    feature_read_workers : int
        Forwarded to :class:`~precisionai.agritune.features.provider.CachedFeatureProvider` — the
        thread-pool size for concurrent per-batch store reads (only consulted under
        ``feature_provider: cached``). Independent of ``num_workers``: that pool decodes/augments
        raw images (or is unused entirely when ``feature_provider: cached`` skips image loading —
        see ``build_training_batches``'s ``load_images``); this one reads already-computed features
        off disk, the dominant remaining per-batch cost once images are skipped.
    val_fraction : float
        Fraction of samples held out for validation (via a random split). Validation always uses
        unaugmented samples, regardless of ``augmentation.mode``.
    seed : int
    optimizer : OptimizerConfig
    scheduler : SchedulerConfig | None
    trainer : TrainerConfig
    loss : SegmentationLossConfig
    val_metric_name : str
    higher_is_better : bool
    checkpoint_top_k : int
        Maximum number of best-by-validation-metric periodic checkpoints to retain; ``0`` keeps
        every one — see :class:`~precisionai.agritune.training.checkpointing.CheckpointManager`.
    tracking : TrackingSelection
    original_config : dict[str, Any]
        Unresolved source mapping captured by the CLI config loader for provenance. Direct Python
        callers may leave it empty, in which case the resolved configuration is recorded instead.
    config_overrides : list[str]
        Hydra/dotlist overrides supplied alongside the source configuration.
    """

    manifest_path: str
    feature_store_dir: str
    run_root: str
    run_id: str
    num_classes: int
    encoder_fingerprint: EncoderFingerprint
    feature_provider: str = "cached"
    store_type: str = "directory"
    entries_per_shard: int = 1000
    encoder_base_url: str | None = None
    encoder_api_key: str | None = None
    augmentation: AugmentationSelection = field(default_factory=AugmentationSelection)
    feature_augmentation: FeatureAugmentationConfig = field(default_factory=FeatureAugmentationConfig)
    decoder_name: str = "mlp_probe"
    decoder_kwargs: dict[str, Any] = field(default_factory=dict)
    batch_size: int = 4
    device: str = "cpu"
    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: int | None = None
    feature_read_workers: int = 32
    val_fraction: float = 0.2
    seed: int = 0
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig | None = None
    trainer: TrainerConfig = field(default_factory=lambda: TrainerConfig(max_epochs=1))
    loss: SegmentationLossConfig = field(default_factory=SegmentationLossConfig)
    val_metric_name: str = "mean_iou"
    higher_is_better: bool = True
    checkpoint_top_k: int = 3
    tracking: TrackingSelection = field(default_factory=TrackingSelection)
    original_config: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    config_overrides: list[str] = field(default_factory=list, repr=False, compare=False)


@dataclass
class TrainingRunResult:
    """What one :func:`run_training` call produced.

    Attributes
    ----------
    run_directory : RunDirectory
    final_train_state : dict[str, Any]
        A plain-dict snapshot of the trainer's final :class:`~precisionai.agritune.training.state.TrainingState`.
    train_metrics : dict[str, float]
        The last training epoch's metrics (see ``Trainer.fit``'s ``train_metric`` parameter).
    val_metrics : dict[str, float]
        The last validation pass's metrics.
    """

    run_directory: RunDirectory
    final_train_state: dict[str, Any]
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]


def _validate_dataloader_config(config: TrainingRunConfig) -> None:
    if config.num_workers < 0:
        raise ValueError(f"num_workers must be non-negative; got {config.num_workers}")
    if config.prefetch_factor is not None and config.num_workers == 0:
        raise ValueError(f"prefetch_factor={config.prefetch_factor} requires num_workers > 0; got num_workers=0")
    if config.prefetch_factor is not None and config.prefetch_factor < 1:
        raise ValueError(f"prefetch_factor must be positive; got {config.prefetch_factor}")


def _validate_feature_store_config(config: TrainingRunConfig) -> None:
    if config.store_type not in FEATURE_STORE_TYPES:
        raise ValueError(f"unsupported store_type: {config.store_type!r}; expected one of {FEATURE_STORE_TYPES}")
    if config.entries_per_shard < 1:
        raise ValueError(f"entries_per_shard must be positive; got {config.entries_per_shard}")
    if config.feature_read_workers < 1:
        raise ValueError(f"feature_read_workers must be positive; got {config.feature_read_workers}")


def _validate_device_config(config: TrainingRunConfig) -> None:
    validate_device(config.device)


def _validate_config(config: TrainingRunConfig) -> None:
    if config.num_classes < 1:
        raise ValueError(f"num_classes must be positive; got {config.num_classes}")
    if config.batch_size < 1:
        raise ValueError(f"batch_size must be positive; got {config.batch_size}")
    _validate_dataloader_config(config)
    if config.trainer.max_epochs < 1:
        raise ValueError(f"trainer.max_epochs must be positive; got {config.trainer.max_epochs}")
    if config.checkpoint_top_k < 0:
        raise ValueError(f"checkpoint_top_k must be non-negative; got {config.checkpoint_top_k}")
    if config.feature_provider not in _FEATURE_PROVIDERS:
        raise ValueError(
            f"unsupported feature_provider: {config.feature_provider!r}; expected one of {_FEATURE_PROVIDERS}"
        )
    _validate_feature_store_config(config)
    _validate_device_config(config)
    online_augmentation = config.augmentation.mode in (AugmentationMode.ONLINE, AugmentationMode.HYBRID)
    if online_augmentation and config.feature_provider == "cached":
        raise ValueError(
            f"invalid combination: augmentation.mode={config.augmentation.mode.value!r} requires "
            "feature_provider 'online' or 'hybrid' — a cached provider's key would only ever match "
            "the first epoch's augmentation fingerprint"
        )


def _build_feature_provider(
    config: TrainingRunConfig, *, store: DirectoryFeatureStore | ShardedFeatureStore
) -> tuple[FeatureProvider, list]:
    if config.feature_provider == "cached":
        return build_cached_feature_provider(
            config.manifest_path,
            store=store,
            encoder_fingerprint=config.encoder_fingerprint,
            max_read_workers=config.feature_read_workers,
        )

    rows = load_manifest(config.manifest_path)
    gateway, _ = build_encoder(
        base_url=config.encoder_base_url,
        api_key=config.encoder_api_key,
        model=config.encoder_fingerprint.model,
        preprocessing=config.encoder_fingerprint.preprocessing,
    )
    if config.feature_provider == "online":
        return OnlineFeatureProvider(gateway), rows

    provider = HybridFeatureProvider(
        store,
        gateway=gateway,
        encoder_fingerprint=config.encoder_fingerprint,
        image_hash_fn=build_image_hash_fn(config.manifest_path),
    )
    return provider, rows


def _build_augmentation_pipeline(config: TrainingRunConfig) -> ImageAugmentationPipeline:
    return ImageAugmentationPipeline(
        AugmentationPipelineConfig(geometric=config.augmentation.geometric, photometric=config.augmentation.photometric)
    )


def _build_train_batches(
    config: TrainingRunConfig, train_dataset: ManifestDataset, *, show_progress: bool = False
) -> Any:
    # A cached provider never consults PreparedSample.image (its features are already computed);
    # only hybrid/online do, to encode fresh on a miss or every epoch respectively — so only cached
    # is safe to skip decoding the image for entirely.
    load_images = config.feature_provider != "cached"
    if config.augmentation.mode is AugmentationMode.NONE:
        # geometric.resize is dimensional normalization, not augmentation — apply it here exactly
        # like val_batches always does below, regardless of mode. Otherwise a dataset whose
        # images/masks are not already one uniform native size fails deep in a DataLoader worker
        # with a confusing torch.stack size-mismatch, even though resize was configured — this
        # never creates an AugmentationRecord (see _ResizedBatches._item), so it does not change
        # the "none" augmentation fingerprint cached features/lookups key off.
        return build_training_batches(
            train_dataset,
            batch_size=config.batch_size,
            resize=config.augmentation.geometric.resize,
            show_progress=show_progress,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
            load_images=load_images,
        )

    pipeline = _build_augmentation_pipeline(config)
    if config.augmentation.mode is AugmentationMode.OFFLINE:
        return build_static_augmented_batches(
            train_dataset,
            pipeline=pipeline,
            mode=AugmentationMode.OFFLINE,
            global_seed=config.seed,
            batch_size=config.batch_size,
            variant=config.augmentation.variant,
            show_progress=show_progress,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
            load_images=load_images,
        )

    return OnlineAugmentedBatches(
        train_dataset,
        pipeline=pipeline,
        mode=config.augmentation.mode,
        global_seed=config.seed,
        batch_size=config.batch_size,
        hybrid_online_probability=config.augmentation.hybrid_online_probability,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
    )


def _close_tracker_and_flush_store(tracker: Tracker, store: DirectoryFeatureStore | ShardedFeatureStore) -> None:
    """Close ``tracker`` and flush ``store``, run unconditionally from ``run_training``'s ``finally``.

    ``tracker.close()`` failing never masks whatever exception is already propagating from
    ``trainer.fit()`` — it is always just logged. ``store.flush()`` gets the same treatment only
    when ``fit()`` already failed; otherwise its own failure is a real error and must propagate
    normally, since nothing else went wrong to explain suppressing it.
    """
    try:
        tracker.close()
    except Exception:
        logger.exception("tracking backend failed while closing; training artifacts remain valid")

    fit_already_failing = sys.exc_info()[0] is not None
    try:
        store.flush()
    except Exception:
        if not fit_already_failing:
            raise
        logger.exception("feature store failed to flush while trainer.fit() was already failing")


def run_training(
    config: TrainingRunConfig, *, store: DirectoryFeatureStore | ShardedFeatureStore, show_progress: bool = False
) -> TrainingRunResult:
    """Run one full training job: split -> features -> (optional augmentation) -> task -> trainer.

    Parameters
    ----------
    config : TrainingRunConfig
        Fully resolved run configuration.
    store : DirectoryFeatureStore | ShardedFeatureStore
        Already exists as a directory; populated already under ``feature_provider: cached``,
        populated incrementally (write-through) under ``"hybrid"``, unused under ``"online"``.
    show_progress : bool, optional
        Render ``tqdm`` bars over training/validation batches — see :class:`Trainer`. Defaults to
        ``False`` so headless callers (e.g. the API) see no terminal output.

    Returns
    -------
    TrainingRunResult

    Raises
    ------
    ValueError
        If ``config.feature_provider`` is not recognized, or ``augmentation.mode`` is
        ``online``/``hybrid`` while ``feature_provider`` is ``cached``.
    """
    _validate_config(config)
    logger.info(
        "starting training run %r (decoder=%s, feature_provider=%s, max_epochs=%d)",
        config.run_id,
        config.decoder_name,
        config.feature_provider,
        config.trainer.max_epochs,
    )
    provider, rows = _build_feature_provider(config, store=store)

    # random_split is a three-way (train/val/test) split; a tiny epsilon is reserved for "test"
    # (unused here) so train_fraction + val_fraction stays strictly below 1, as it requires.
    train_fraction = max(1e-6, 1.0 - config.val_fraction - 1e-6)
    split: SplitAssignment = random_split(
        rows, val_fraction=config.val_fraction, train_fraction=train_fraction, seed=config.seed
    )
    if not split.train:
        raise ValueError("training split is empty; adjust val_fraction/seed or add more samples")
    dataset_fingerprint = _dataset_fingerprint(config.manifest_path, rows, show_progress=show_progress)
    split_fingerprint = _stable_fingerprint(asdict(split))
    encoder_fingerprint = _stable_fingerprint(
        {"encoder": asdict(config.encoder_fingerprint), "base_url": config.encoder_base_url}
    )
    critical_fingerprints = {
        "config": _stable_fingerprint(_critical_config(config)),
        "dataset": dataset_fingerprint,
        "split": split_fingerprint,
        "encoder": encoder_fingerprint,
    }

    train_dataset = ManifestDataset(config.manifest_path, sample_ids=split.train)
    val_dataset = ManifestDataset(config.manifest_path, sample_ids=split.val or split.train)

    train_batches = _build_train_batches(config, train_dataset, show_progress=show_progress)
    # Validation is never augmented (see docs/augmentation.md), but it still needs every sample's
    # image/mask resized to one common size before torch.stack — so it reuses the configured
    # geometric.resize (deterministic dimensional normalization, not augmentation) regardless of
    # config.augmentation.mode.
    val_batches = build_training_batches(
        val_dataset,
        batch_size=config.batch_size,
        resize=config.augmentation.geometric.resize,
        show_progress=show_progress,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
        load_images=config.feature_provider != "cached",
    )

    first_train_sample = train_dataset[0]
    prepared_probe = prepare_sample(
        first_train_sample,
        mode=config.augmentation.mode,
        pipeline=_build_augmentation_pipeline(config),
        global_seed=config.seed,
        variant=config.augmentation.variant,
        epoch=0,
        hybrid_online_probability=config.augmentation.hybrid_online_probability,
    )
    probe_sample = PreparedSample(
        sample_id=prepared_probe.sample_id,
        image=prepared_probe.image,
        target=None,
        augmentation_metadata=prepared_probe.augmentation_metadata,
    )
    patch_dim, cls_dim = probe_feature_dims(provider, probe_sample)
    # From the *augmented* target, not first_train_sample.target directly: geometric augmentation
    # (resize/random_crop) changes the spatial size every training batch's target actually has, so
    # probing the pre-augmentation sample would build a decoder upsampling to the wrong resolution.
    output_size = mask_output_size(prepared_probe.target)

    decoder = build_decoder(
        config.decoder_name,
        patch_dim=patch_dim,
        cls_dim=cls_dim,
        num_classes=config.num_classes,
        output_size=output_size,
        **config.decoder_kwargs,
    )
    decoder = decoder.to(torch.device(config.device))
    task = SegmentationTask(decoder, SegmentationLoss(config.loss, num_classes=config.num_classes))
    optimizer = build_optimizer(decoder.parameters(), config.optimizer)
    scheduler = build_scheduler(optimizer, config.scheduler) if config.scheduler is not None else None

    run_dir = RunDirectory(config.run_root, config.run_id)
    checkpoint_manager = CheckpointManager(run_dir.checkpoints_dir, top_k=config.checkpoint_top_k)
    tracker = build_trackers(config.tracking, run_dir=run_dir, run_id=config.run_id)

    trainer_config = replace(
        config.trainer,
        fingerprints={**config.trainer.fingerprints, **critical_fingerprints},
    )
    trainer = Trainer(
        task=task,
        decoder=decoder,
        feature_provider=provider,
        optimizer=optimizer,
        scheduler=scheduler,
        config=trainer_config,
        checkpoint_manager=checkpoint_manager,
        tracker=tracker,
        feature_augmentation=FeatureAugmentationPipeline(config.feature_augmentation),
        show_progress=show_progress,
    )

    if isinstance(train_batches, OnlineAugmentedBatches):
        train_batches.set_epoch(trainer.state.epoch)

    train_metric = SegmentationMetric(num_classes=config.num_classes)
    val_metric = SegmentationMetric(num_classes=config.num_classes)
    try:
        trainer.fit(
            train_batches,
            val_batches,
            train_metric=train_metric,
            val_metric=val_metric,
            val_metric_name=config.val_metric_name,
            higher_is_better=config.higher_is_better,
        )
        final_train_metrics = (
            trainer.last_train_metrics if trainer.last_train_metrics is not None else train_metric.compute()
        )
        final_val_metrics = trainer.last_val_metrics if trainer.last_val_metrics is not None else val_metric.compute()
    finally:
        _close_tracker_and_flush_store(tracker, store)

    feature_cache_fingerprint = _stable_fingerprint(sorted(store.list_keys()))
    run_dir.write_provenance(
        config_original=_original_config(config),
        config_resolved=_resolved_config(config),
        run_info={
            "seed": config.seed,
            "epoch": trainer.state.epoch,
            "global_optimizer_step": trainer.state.global_optimizer_step,
            "best_metric": trainer.state.best_metric,
            "config_fingerprint": critical_fingerprints["config"],
            "dataset_fingerprint": dataset_fingerprint,
            "split_fingerprint": split_fingerprint,
            "encoder_fingerprint": encoder_fingerprint,
            "feature_cache_fingerprint": feature_cache_fingerprint,
        },
        dataset_info={
            "manifest_path": config.manifest_path,
            "fingerprint": dataset_fingerprint,
            "split_fingerprint": split_fingerprint,
            "split": asdict(split),
        },
        encoder_info={
            "model": config.encoder_fingerprint.model,
            "revision": config.encoder_fingerprint.revision,
            "preprocessing": config.encoder_fingerprint.preprocessing,
            "fingerprint": encoder_fingerprint,
        },
    )

    logger.info(
        "finished training run %r at epoch %d: run directory %s",
        config.run_id,
        trainer.state.epoch,
        run_dir.path,
    )
    return TrainingRunResult(
        run_directory=run_dir,
        final_train_state={
            "epoch": trainer.state.epoch,
            "global_optimizer_step": trainer.state.global_optimizer_step,
            "best_metric": trainer.state.best_metric,
        },
        train_metrics=final_train_metrics,
        val_metrics=final_val_metrics,
    )
