# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Prediction orchestration — powers ``agritune predict``.

Loads a trained decoder from a checkpoint and writes one per-pixel class-index PNG per sample.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from precisionai.agritune.augmentations.image.pipeline import (
    AugmentationMode,
    AugmentationPipelineConfig,
    GeometricConfig,
    ImageAugmentationPipeline,
    prepare_sample,
)
from precisionai.agritune.data.dataset import ManifestDataset
from precisionai.agritune.features.keys import EncoderFingerprint
from precisionai.agritune.logging import get_logger, progress_iter
from precisionai.agritune.schemas.protocols import FeatureStore
from precisionai.agritune.schemas.samples import PreparedSample
from precisionai.agritune.services.segmentation_common import (
    build_cached_feature_provider,
    build_decoder,
    build_prediction_batches,
    build_static_augmented_batches,
    mask_output_size,
    probe_feature_dims,
    validate_device,
)
from precisionai.agritune.tasks.segmentation.postprocessing import logits_to_predictions
from precisionai.agritune.tasks.segmentation.visualization import overlay_predictions_on_image
from precisionai.agritune.training.checkpointing import CheckpointManager

logger = get_logger(__name__)


@dataclass
class PredictionRunConfig:
    """Configuration for one :func:`run_prediction` call.

    Attributes
    ----------
    manifest_path : str
        Dataset manifest CSV (its mask column supplies the output resolution; predictions do not
        require accurate ground truth, only correctly shaped entries).
    checkpoint_path : str
        Path to a checkpoint written by :mod:`training_service`.
    output_dir : str
        Directory predictions are written into (typically a run's ``predictions/`` directory).
    num_classes : int
    encoder_fingerprint : EncoderFingerprint
    decoder_name : str
        ``"mlp_probe"``, ``"token_fpn"``, ``"aspp"``, ``"ppm"``, ``"segmenter"``, or
        ``"mask_former"`` — must match the checkpointed decoder's architecture.
    decoder_kwargs : dict[str, Any]
        Extra keyword arguments the decoder was constructed with during training (e.g.
        ``hidden_dims``) — must match exactly, or the checkpoint's state dict will not load.
    batch_size : int
    sample_ids : list[str] | None
        Restrict to these sample IDs; ``None`` predicts every row in the manifest.
    device : str
        Where the decoder and every batch's features are moved before inference — ``"cpu"`` (the
        default), ``"cuda"``, or a specific GPU like ``"cuda:3"``. Rejected up front if it names a
        CUDA device that either isn't available at all or is out of range for this machine's GPU
        count — this never silently falls back to CPU. Independent of whatever ``device`` the
        checkpoint was trained under.
    resize : tuple[int, int] | None
        ``(width, height)`` every image is deterministically resized to before batching — must
        match the checkpoint's training run (its ``augmentation.geometric.resize``). Required
        whenever the dataset's images do not already share one native size. Ignored under
        ``augmentation_mode: offline``, whose own ``augmentation_pipeline`` already includes
        whatever resize the training run used.
    write_overlays : bool
        Also write ``{sample_id}_overlay.png``: the original image with the predicted mask
        alpha-blended on top, for quick visual review — see
        :mod:`~precisionai.agritune.tasks.segmentation.visualization`.
    overlay_alpha : float
        Overlay opacity in ``[0, 1]``, passed to
        :func:`~precisionai.agritune.tasks.segmentation.visualization.overlay_predictions_on_image`.
        Ignored unless ``write_overlays`` is set.
    augmentation_mode : AugmentationMode
        ``NONE`` (default) predicts on each sample's native, unaugmented image — the original
        behavior. ``OFFLINE`` instead reapplies the same deterministic ``augmentation_pipeline``
        (typically a ``resize``/``random_crop``) the checkpoint's decoder was actually trained
        under, and reads features from the matching offline-augmented cache entry rather than the
        native one. Required whenever training used ``augmentation.mode: offline`` with a
        ``random_crop``: a decoder trained only on small, fixed-size crops has no spatial context
        beyond a single patch (see ``mlp_probe``) and generalizes poorly to a much larger, native,
        differently-shaped patch grid it never saw during training — this is not merely lower
        accuracy, it can produce content-independent, purely position-driven artifacts. ``ONLINE``
        and ``HYBRID`` are not supported (predicting against a live-changing augmentation makes no
        sense for a fixed inference pass).
    augmentation_pipeline : ImageAugmentationPipeline
        Ignored under ``NONE``. Must be built from the exact same config as the training run's
        ``augmentation:`` block (e.g. via ``agritune features build``'s own
        ``--augmentation-config`` file), or the derived cache key won't match what was cached.
    global_seed : int
        Ignored under ``NONE``. Must match the training run's seed, or the derived per-sample
        augmentation (and so the cache key) won't match.
    augmentation_variant : int
        Ignored under ``NONE``. Must match the training run's offline variant index.
    """

    manifest_path: str
    checkpoint_path: str
    output_dir: str
    num_classes: int
    encoder_fingerprint: EncoderFingerprint
    decoder_name: str = "mlp_probe"
    decoder_kwargs: dict[str, Any] = field(default_factory=dict)
    batch_size: int = 4
    sample_ids: list[str] | None = None
    device: str = "cpu"
    resize: tuple[int, int] | None = None
    write_overlays: bool = False
    overlay_alpha: float = 0.5
    augmentation_mode: AugmentationMode = AugmentationMode.NONE
    augmentation_pipeline: ImageAugmentationPipeline = field(default_factory=ImageAugmentationPipeline)
    global_seed: int = 0
    augmentation_variant: int = 0

    def __post_init__(self) -> None:
        """Reject augmentation modes a fixed inference pass cannot meaningfully use, and validate ``device``."""
        validate_device(self.device)
        if self.augmentation_mode not in (AugmentationMode.NONE, AugmentationMode.OFFLINE):
            raise ValueError(
                f"prediction supports augmentation_mode 'none' or 'offline'; got {self.augmentation_mode.value!r}"
            )


def _build_sample_batches(
    dataset: ManifestDataset, config: PredictionRunConfig, *, show_progress: bool
) -> Iterable[Sequence[PreparedSample]]:
    """Return the per-batch sample lists ``run_prediction`` reads features and images from."""
    if config.augmentation_mode is AugmentationMode.NONE:
        return build_prediction_batches(dataset, batch_size=config.batch_size, resize=config.resize)
    return (
        batch.samples
        for batch in build_static_augmented_batches(
            dataset,
            pipeline=config.augmentation_pipeline,
            mode=config.augmentation_mode,
            global_seed=config.global_seed,
            batch_size=config.batch_size,
            variant=config.augmentation_variant,
            show_progress=show_progress,
        )
    )


def run_prediction(config: PredictionRunConfig, *, store: FeatureStore, show_progress: bool = False) -> list[str]:
    """Run inference and write one prediction PNG per sample.

    Parameters
    ----------
    config : PredictionRunConfig
    store : FeatureStore
        An already-populated store (see :mod:`~precisionai.agritune.services.feature_service`).
    show_progress : bool, optional
        Render a ``tqdm`` bar over prediction batches. Defaults to ``False`` so headless callers
        (e.g. the API) see no terminal output.

    Returns
    -------
    list[str]
        Paths written, one per sample, in dataset order.

    Raises
    ------
    ValueError
        If there are no samples to predict on.
    """
    provider, _ = build_cached_feature_provider(
        config.manifest_path, store=store, encoder_fingerprint=config.encoder_fingerprint
    )
    dataset = ManifestDataset(config.manifest_path, sample_ids=config.sample_ids)
    if len(dataset) == 0:
        raise ValueError("no samples to predict")
    logger.info("predicting with checkpoint %s over %d sample(s)", config.checkpoint_path, len(dataset))

    first_sample = dataset[0]
    if config.augmentation_mode is AugmentationMode.NONE:
        probe_sample = PreparedSample(sample_id=first_sample.sample_id, image=None, target=None)
        if config.resize is not None:
            resize_pipeline = ImageAugmentationPipeline(
                AugmentationPipelineConfig(geometric=GeometricConfig(resize=config.resize))
            )
            # From the resized target, not first_sample.target directly: config.resize (when set)
            # changes the spatial size every prediction batch's image actually has, so probing the
            # pre-resize sample would build a decoder upsampling to the wrong resolution — see
            # training_service.run_training's identical probe for why.
            output_size = mask_output_size(resize_pipeline.apply(first_sample, seed=0).target)
        else:
            output_size = mask_output_size(first_sample.target)
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
            image=prepared_probe.image,
            target=None,
            augmentation_metadata=prepared_probe.augmentation_metadata,
        )
        # From the augmented target, not first_sample.target directly: the augmentation pipeline
        # (resize/random_crop) changes the spatial size every batch's samples actually have, so
        # probing the pre-augmentation sample would build a decoder upsampling to the wrong
        # resolution — see training_service.run_training's identical probe for why.
        output_size = mask_output_size(prepared_probe.target)
    patch_dim, cls_dim = probe_feature_dims(provider, probe_sample)

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

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[str] = []

    batches = _build_sample_batches(dataset, config, show_progress=show_progress)
    total_batches = -(-len(dataset) // config.batch_size)
    with torch.no_grad():
        for batch in progress_iter(
            batches, desc="predict", unit="batch", disable=not show_progress, total=total_batches
        ):
            features = provider.get_features(batch).to(device)
            predictions = logits_to_predictions(decoder(features)).cpu()
            for sample, prediction in zip(batch, predictions, strict=True):
                path = output_dir / f"{sample.sample_id}.png"
                Image.fromarray(prediction.numpy().astype(np.uint8)).save(path)
                written_paths.append(str(path))

                if config.write_overlays:
                    overlay_path = output_dir / f"{sample.sample_id}_overlay.png"
                    overlay = overlay_predictions_on_image(
                        sample.image, prediction, num_classes=config.num_classes, alpha=config.overlay_alpha
                    )
                    overlay.save(overlay_path)
                    written_paths.append(str(overlay_path))

    logger.info("wrote %d file(s) to %s", len(written_paths), output_dir)
    return written_paths
