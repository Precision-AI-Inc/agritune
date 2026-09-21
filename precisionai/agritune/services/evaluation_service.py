# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Evaluation orchestration — powers ``agritune evaluate``.

Loads a trained decoder from a checkpoint and scores it against a dataset (or a restricted set of
sample IDs, e.g. a held-out test split).
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from precisionai.agritune.augmentations.image.pipeline import (
    AugmentationMode,
    ImageAugmentationPipeline,
    prepare_sample,
)
from precisionai.agritune.data.dataset import ManifestDataset
from precisionai.agritune.features.keys import EncoderFingerprint
from precisionai.agritune.logging import get_logger
from precisionai.agritune.schemas.protocols import FeatureStore
from precisionai.agritune.schemas.samples import PreparedSample
from precisionai.agritune.services.segmentation_common import (
    build_cached_feature_provider,
    build_decoder,
    build_static_augmented_batches,
    build_training_batches,
    mask_output_size,
    probe_feature_dims,
    validate_device,
)
from precisionai.agritune.tasks.segmentation.losses import SegmentationLoss, SegmentationLossConfig
from precisionai.agritune.tasks.segmentation.metrics import SegmentationMetric
from precisionai.agritune.tasks.segmentation.task import SegmentationTask
from precisionai.agritune.training.checkpointing import CheckpointManager
from precisionai.agritune.training.evaluator import TrainingBatch, evaluate

logger = get_logger(__name__)


@dataclass
class EvaluationRunConfig:
    """Configuration for one :func:`run_evaluation` call.

    Attributes
    ----------
    manifest_path : str
        Dataset manifest CSV.
    checkpoint_path : str
        Path to a checkpoint written by :mod:`training_service` (e.g. ``last.ckpt``/``best.ckpt``).
    num_classes : int
    encoder_fingerprint : EncoderFingerprint
        Must match what the feature store was built with.
    decoder_name : str
        ``"mlp_probe"``, ``"token_fpn"``, ``"aspp"``, ``"ppm"``, ``"segmenter"``, or
        ``"mask_former"`` — must match the checkpointed decoder's architecture.
    decoder_kwargs : dict[str, Any]
        Extra keyword arguments the decoder was constructed with during training (e.g.
        ``hidden_dims``) — must match exactly, or the checkpoint's state dict will not load.
    batch_size : int
    sample_ids : list[str] | None
        Restrict evaluation to these sample IDs (e.g. a held-out test split); ``None`` evaluates
        every row in the manifest.
    device : str
        Where the decoder and every batch's features/targets are moved before evaluation —
        ``"cpu"`` (the default), ``"cuda"``, or a specific GPU like ``"cuda:3"``. Rejected up front
        if it names a CUDA device that either isn't available at all or is out of range for this
        machine's GPU count — this never silently falls back to CPU. Independent of whatever
        ``device`` the checkpoint was trained under.
    loss : SegmentationLossConfig
        Should match the loss the checkpoint was trained under, or the reported ``"loss"`` value
        won't be comparable to training/validation loss from that run.
    resize : tuple[int, int] | None
        ``(width, height)`` every sample's image and mask are deterministically resized to before
        batching. Must match whatever the checkpoint's training run resized to (its
        ``augmentation.geometric.resize``), and is required whenever the dataset's own
        images/masks do not already share one native size — otherwise batching them fails. Ignored
        under ``augmentation_mode: offline``, where ``augmentation_pipeline`` governs sizing.
    augmentation_mode : AugmentationMode
        ``NONE`` (default) evaluates on each sample's native, unaugmented image (optionally
        resized via ``resize``) — the original behavior. ``OFFLINE`` instead reapplies the same
        deterministic ``augmentation_pipeline`` (typically a ``resize``/``random_crop``) the
        checkpoint's decoder was actually trained under, and reads features from the matching
        offline-augmented cache entry rather than the native one. Required whenever training used
        ``augmentation.mode: offline`` with a ``random_crop``: a decoder trained only on small,
        fixed-size crops has no spatial context beyond a single patch (e.g. ``mlp_probe``) and
        generalizes poorly to a much larger, native, differently-shaped patch grid it never saw
        during training — see :mod:`~precisionai.agritune.services.prediction_service` for the
        identical concern on the predict side. ``ONLINE``/``HYBRID`` are not supported (evaluating
        against a live-changing augmentation makes no sense for a fixed, reproducible pass).
    augmentation_pipeline : ImageAugmentationPipeline
        Ignored under ``NONE``. Must be built from the exact same config as the training run's
        ``augmentation:`` block, or the derived cache key won't match what was cached.
    global_seed : int
        Ignored under ``NONE``. Must match the training run's seed, or the derived per-sample
        augmentation (and so the cache key) won't match.
    augmentation_variant : int
        Ignored under ``NONE``. Must match the training run's offline variant index.
    """

    manifest_path: str
    checkpoint_path: str
    num_classes: int
    encoder_fingerprint: EncoderFingerprint
    decoder_name: str = "mlp_probe"
    decoder_kwargs: dict[str, Any] = field(default_factory=dict)
    batch_size: int = 4
    sample_ids: list[str] | None = None
    device: str = "cpu"
    loss: SegmentationLossConfig = field(default_factory=SegmentationLossConfig)
    resize: tuple[int, int] | None = None
    augmentation_mode: AugmentationMode = AugmentationMode.NONE
    augmentation_pipeline: ImageAugmentationPipeline = field(default_factory=ImageAugmentationPipeline)
    global_seed: int = 0
    augmentation_variant: int = 0

    def __post_init__(self) -> None:
        """Reject augmentation modes a fixed evaluation pass cannot meaningfully use, and validate ``device``."""
        if self.augmentation_mode not in (AugmentationMode.NONE, AugmentationMode.OFFLINE):
            raise ValueError(
                f"evaluation supports augmentation_mode 'none' or 'offline'; got {self.augmentation_mode.value!r}"
            )
        validate_device(self.device)


def _build_batches(
    dataset: ManifestDataset, config: EvaluationRunConfig, *, show_progress: bool
) -> Iterable[TrainingBatch]:
    """Return the batches ``run_evaluation`` reads features and targets from."""
    if config.augmentation_mode is AugmentationMode.NONE:
        return build_training_batches(
            dataset, batch_size=config.batch_size, resize=config.resize, show_progress=show_progress
        )
    return build_static_augmented_batches(
        dataset,
        pipeline=config.augmentation_pipeline,
        mode=config.augmentation_mode,
        global_seed=config.global_seed,
        batch_size=config.batch_size,
        variant=config.augmentation_variant,
        show_progress=show_progress,
    )


def run_evaluation(
    config: EvaluationRunConfig, *, store: FeatureStore, show_progress: bool = False
) -> dict[str, float]:
    """Load a checkpointed decoder and evaluate it over a dataset.

    Parameters
    ----------
    config : EvaluationRunConfig
    store : FeatureStore
        An already-populated store (see :mod:`~precisionai.agritune.services.feature_service`).
    show_progress : bool, optional
        Render a ``tqdm`` bar over evaluation batches. Defaults to ``False`` so headless callers
        (e.g. the API) see no terminal output.

    Returns
    -------
    dict[str, float]
        The metrics :meth:`~precisionai.agritune.tasks.segmentation.metrics.SegmentationMetric.compute`
        reports, plus a ``"loss"`` entry computed under ``config.loss``
        (see :func:`~precisionai.agritune.training.evaluator.evaluate`).

    Raises
    ------
    ValueError
        If no samples match ``config.sample_ids`` (or the manifest is empty).
    """
    provider, _ = build_cached_feature_provider(
        config.manifest_path, store=store, encoder_fingerprint=config.encoder_fingerprint
    )
    dataset = ManifestDataset(config.manifest_path, sample_ids=config.sample_ids)
    if len(dataset) == 0:
        raise ValueError("no samples to evaluate")
    logger.info("evaluating checkpoint %s over %d sample(s)", config.checkpoint_path, len(dataset))

    batches = _build_batches(dataset, config, show_progress=show_progress)

    first_sample = dataset[0]
    if config.augmentation_mode is AugmentationMode.NONE:
        probe_sample = PreparedSample(sample_id=first_sample.sample_id, image=None, target=None)
    else:
        prepared_probe = prepare_sample(
            first_sample,
            mode=config.augmentation_mode,
            pipeline=config.augmentation_pipeline,
            global_seed=config.global_seed,
            variant=config.augmentation_variant,
        )
        probe_sample = PreparedSample(
            sample_id=prepared_probe.sample_id,
            image=None,
            target=None,
            augmentation_metadata=prepared_probe.augmentation_metadata,
        )
    patch_dim, cls_dim = probe_feature_dims(provider, probe_sample)
    # From the first prepared batch, not first_sample.target directly: config.resize (when set)
    # changes the spatial size every evaluation batch's target actually has, so probing the
    # pre-resize sample would build a decoder upsampling to the wrong resolution. `batches` is a
    # lazily re-iterable object (not a list), so this reads one extra chunk from disk rather than
    # indexing — negligible next to the dataset-wide pass `evaluate()` makes below.
    output_size = mask_output_size(next(iter(batches)).targets[0])

    decoder = build_decoder(
        config.decoder_name,
        patch_dim=patch_dim,
        cls_dim=cls_dim,
        num_classes=config.num_classes,
        output_size=output_size,
        **config.decoder_kwargs,
    )
    checkpoint = CheckpointManager(Path(config.checkpoint_path).parent).load(config.checkpoint_path)
    decoder.load_state_dict(checkpoint.decoder_state)
    device = torch.device(config.device)
    decoder = decoder.to(device)
    decoder.eval()

    task = SegmentationTask(decoder, SegmentationLoss(config.loss, num_classes=config.num_classes))
    metric = SegmentationMetric(num_classes=config.num_classes)
    return evaluate(task, provider, batches, metric, device=device, show_progress=show_progress)
