# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Prediction orchestration — powers ``agritune predict``.

Loads a trained decoder from a checkpoint and writes one per-pixel class-index PNG per sample.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from precisionai.agritune.data.dataset import ManifestDataset
from precisionai.agritune.features.keys import EncoderFingerprint
from precisionai.agritune.logging import get_logger, progress_iter
from precisionai.agritune.schemas.protocols import FeatureStore
from precisionai.agritune.schemas.samples import PreparedSample
from precisionai.agritune.services.segmentation_common import (
    build_cached_feature_provider,
    build_decoder,
    build_prediction_batches,
    mask_output_size,
    probe_feature_dims,
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
    write_overlays : bool
        Also write ``{sample_id}_overlay.png``: the original image with the predicted mask
        alpha-blended on top, for quick visual review — see
        :mod:`~precisionai.agritune.tasks.segmentation.visualization`.
    overlay_alpha : float
        Overlay opacity in ``[0, 1]``, passed to
        :func:`~precisionai.agritune.tasks.segmentation.visualization.overlay_predictions_on_image`.
        Ignored unless ``write_overlays`` is set.
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
    write_overlays: bool = False
    overlay_alpha: float = 0.5


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
    probe_sample = PreparedSample(sample_id=first_sample.sample_id, image=None, target=None)
    patch_dim, cls_dim = probe_feature_dims(provider, probe_sample)
    output_size = mask_output_size(first_sample.target)

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
    decoder.eval()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[str] = []

    batches = build_prediction_batches(dataset, batch_size=config.batch_size)
    with torch.no_grad():
        for batch in progress_iter(batches, desc="predict", unit="batch", disable=not show_progress):
            features = provider.get_features(batch)
            predictions = logits_to_predictions(decoder(features))
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
