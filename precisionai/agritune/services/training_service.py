# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Training orchestration — wires dataset + feature store + task + trainer into ``agritune train``.

Supports every ``augmentation.mode`` x ``feature_provider`` combination the plan's config
hierarchy (§23) calls for, except ``cached`` combined with ``online``/``hybrid`` augmentation —
rejected up front, since a cached provider's key would only ever match the first epoch's
augmentation fingerprint. This is also the module the plan's central integration test exercises:
tiny dataset -> fake encoder -> feature precompute -> linear decoder -> train -> checkpoint ->
resume -> evaluate.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from precisionai.agritune.augmentations.image.pipeline import (
    AugmentationMode,
    AugmentationPipelineConfig,
    GeometricConfig,
    ImageAugmentationPipeline,
    PhotometricConfig,
)
from precisionai.agritune.data.dataset import ManifestDataset
from precisionai.agritune.data.manifest import load_manifest
from precisionai.agritune.data.split import SplitAssignment, random_split
from precisionai.agritune.features.keys import EncoderFingerprint
from precisionai.agritune.features.provider import HybridFeatureProvider, OnlineFeatureProvider
from precisionai.agritune.features.store import DirectoryFeatureStore, ShardedFeatureStore
from precisionai.agritune.logging import RunDirectory
from precisionai.agritune.optimization.optimizers import OptimizerConfig, build_optimizer
from precisionai.agritune.optimization.schedulers import SchedulerConfig, build_scheduler
from precisionai.agritune.schemas.protocols import FeatureProvider
from precisionai.agritune.schemas.samples import PreparedSample
from precisionai.agritune.services.encoder_selection import build_encoder
from precisionai.agritune.services.segmentation_common import (
    OnlineAugmentedBatches,
    build_cached_feature_provider,
    build_decoder,
    build_image_hash_fn,
    build_static_augmented_batches,
    build_training_batches,
    probe_feature_dims,
)
from precisionai.agritune.services.tracking_selection import TrackingSelection, build_trackers
from precisionai.agritune.tasks.segmentation.losses import SegmentationLoss, SegmentationLossConfig
from precisionai.agritune.tasks.segmentation.metrics import SegmentationMetric
from precisionai.agritune.tasks.segmentation.task import SegmentationTask
from precisionai.agritune.training.checkpointing import CheckpointManager
from precisionai.agritune.training.trainer import Trainer, TrainerConfig

_FEATURE_PROVIDERS = ("cached", "online", "hybrid")


@dataclass
class AugmentationSelection:
    """Which image-augmentation mode to apply to training samples, and its transform config.

    Attributes
    ----------
    mode : AugmentationMode
        ``NONE`` (default — matches every pre-v0.2 behavior exactly, no augmentation applied),
        ``OFFLINE``, ``ONLINE``, or ``HYBRID``.
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
        Directory a :class:`~precisionai.agritune.features.store.DirectoryFeatureStore` was
        already built into (via ``agritune features build`` / :mod:`feature_service`) — read from
        under ``feature_provider: cached``/``hybrid``, written to (write-through) under ``hybrid``,
        unused under ``online``.
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
    decoder_name : str
        ``"linear"`` or ``"token_fpn"``.
    batch_size : int
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
    """

    manifest_path: str
    feature_store_dir: str
    run_root: str
    run_id: str
    num_classes: int
    encoder_fingerprint: EncoderFingerprint
    feature_provider: str = "cached"
    encoder_base_url: str | None = None
    encoder_api_key: str | None = None
    augmentation: AugmentationSelection = field(default_factory=AugmentationSelection)
    decoder_name: str = "linear"
    batch_size: int = 4
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


@dataclass
class TrainingRunResult:
    """What one :func:`run_training` call produced.

    Attributes
    ----------
    run_directory : RunDirectory
    final_train_state : dict[str, Any]
        A plain-dict snapshot of the trainer's final :class:`~precisionai.agritune.training.state.TrainingState`.
    val_metrics : dict[str, float]
        The last validation pass's metrics.
    """

    run_directory: RunDirectory
    final_train_state: dict[str, Any]
    val_metrics: dict[str, float]


def _validate_config(config: TrainingRunConfig) -> None:
    if config.feature_provider not in _FEATURE_PROVIDERS:
        raise ValueError(
            f"unsupported feature_provider: {config.feature_provider!r}; expected one of {_FEATURE_PROVIDERS}"
        )
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
            config.manifest_path, store=store, encoder_fingerprint=config.encoder_fingerprint
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


def _build_train_batches(config: TrainingRunConfig, train_dataset: ManifestDataset) -> Any:
    if config.augmentation.mode is AugmentationMode.NONE:
        return build_training_batches(train_dataset, batch_size=config.batch_size)

    pipeline = ImageAugmentationPipeline(
        AugmentationPipelineConfig(geometric=config.augmentation.geometric, photometric=config.augmentation.photometric)
    )
    if config.augmentation.mode is AugmentationMode.OFFLINE:
        return build_static_augmented_batches(
            train_dataset,
            pipeline=pipeline,
            mode=AugmentationMode.OFFLINE,
            global_seed=config.seed,
            batch_size=config.batch_size,
            variant=config.augmentation.variant,
        )

    return OnlineAugmentedBatches(
        train_dataset,
        pipeline=pipeline,
        mode=config.augmentation.mode,
        global_seed=config.seed,
        batch_size=config.batch_size,
        hybrid_online_probability=config.augmentation.hybrid_online_probability,
    )


def run_training(config: TrainingRunConfig, *, store: DirectoryFeatureStore | ShardedFeatureStore) -> TrainingRunResult:
    """Run one full training job: split -> features -> (optional augmentation) -> task -> trainer.

    Parameters
    ----------
    config : TrainingRunConfig
        Fully resolved run configuration.
    store : DirectoryFeatureStore | ShardedFeatureStore
        Already exists as a directory; populated already under ``feature_provider: cached``,
        populated incrementally (write-through) under ``"hybrid"``, unused under ``"online"``.

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
    provider, rows = _build_feature_provider(config, store=store)

    # random_split is a three-way (train/val/test) split; a tiny epsilon is reserved for "test"
    # (unused here) so train_fraction + val_fraction stays strictly below 1, as it requires.
    train_fraction = max(1e-6, 1.0 - config.val_fraction - 1e-6)
    split: SplitAssignment = random_split(
        rows, val_fraction=config.val_fraction, train_fraction=train_fraction, seed=config.seed
    )

    train_dataset = ManifestDataset(config.manifest_path, sample_ids=split.train)
    val_dataset = ManifestDataset(config.manifest_path, sample_ids=split.val or split.train)

    train_batches = _build_train_batches(config, train_dataset)
    val_batches = build_training_batches(val_dataset, batch_size=config.batch_size)

    first_train_sample = train_dataset[0]
    probe_sample = PreparedSample(sample_id=first_train_sample.sample_id, image=first_train_sample.image, target=None)
    patch_dim, cls_dim = probe_feature_dims(provider, probe_sample)
    output_size = tuple(np.array(first_train_sample.target).shape)

    decoder = build_decoder(
        config.decoder_name,
        patch_dim=patch_dim,
        cls_dim=cls_dim,
        num_classes=config.num_classes,
        output_size=output_size,
    )
    task = SegmentationTask(decoder, SegmentationLoss(config.loss, num_classes=config.num_classes))
    optimizer = build_optimizer(decoder.parameters(), config.optimizer)
    scheduler = build_scheduler(optimizer, config.scheduler) if config.scheduler is not None else None

    run_dir = RunDirectory(config.run_root, config.run_id)
    checkpoint_manager = CheckpointManager(run_dir.checkpoints_dir, top_k=config.checkpoint_top_k)
    tracker = build_trackers(config.tracking, run_dir=run_dir, run_id=config.run_id)

    trainer = Trainer(
        task=task,
        decoder=decoder,
        feature_provider=provider,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config.trainer,
        checkpoint_manager=checkpoint_manager,
        tracker=tracker,
    )

    val_metric = SegmentationMetric(num_classes=config.num_classes)
    trainer.fit(
        train_batches,
        val_batches,
        val_metric=val_metric,
        val_metric_name=config.val_metric_name,
        higher_is_better=config.higher_is_better,
    )
    final_val_metrics = val_metric.compute()
    tracker.close()

    run_dir.write_provenance(
        config_original={},
        config_resolved={
            "num_classes": config.num_classes,
            "decoder": config.decoder_name,
            "feature_provider": config.feature_provider,
            "augmentation_mode": config.augmentation.mode.value,
            "tracking_backends": config.tracking.backends,
        },
        run_info={
            "seed": config.seed,
            "epoch": trainer.state.epoch,
            "global_optimizer_step": trainer.state.global_optimizer_step,
            "best_metric": trainer.state.best_metric,
        },
        encoder_info={"model": config.encoder_fingerprint.model, "revision": config.encoder_fingerprint.revision},
    )

    return TrainingRunResult(
        run_directory=run_dir,
        final_train_state={
            "epoch": trainer.state.epoch,
            "global_optimizer_step": trainer.state.global_optimizer_step,
            "best_metric": trainer.state.best_metric,
        },
        val_metrics=final_val_metrics,
    )
