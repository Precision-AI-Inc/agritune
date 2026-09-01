# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Evaluation orchestration — powers ``agritune evaluate``.

Loads a trained decoder from a checkpoint and scores it against a dataset (or a restricted set of
sample IDs, e.g. a held-out test split).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from precisionai.agritune.data.dataset import ManifestDataset
from precisionai.agritune.features.keys import EncoderFingerprint
from precisionai.agritune.schemas.protocols import FeatureStore
from precisionai.agritune.schemas.samples import PreparedSample
from precisionai.agritune.services.segmentation_common import (
    build_cached_feature_provider,
    build_decoder,
    build_training_batches,
    probe_feature_dims,
)
from precisionai.agritune.tasks.segmentation.losses import SegmentationLoss, SegmentationLossConfig
from precisionai.agritune.tasks.segmentation.metrics import SegmentationMetric
from precisionai.agritune.tasks.segmentation.task import SegmentationTask
from precisionai.agritune.training.checkpointing import CheckpointManager
from precisionai.agritune.training.evaluator import evaluate


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
    loss : SegmentationLossConfig
        Should match the loss the checkpoint was trained under, or the reported ``"loss"`` value
        won't be comparable to training/validation loss from that run.
    """

    manifest_path: str
    checkpoint_path: str
    num_classes: int
    encoder_fingerprint: EncoderFingerprint
    decoder_name: str = "mlp_probe"
    decoder_kwargs: dict[str, Any] = field(default_factory=dict)
    batch_size: int = 4
    sample_ids: list[str] | None = None
    loss: SegmentationLossConfig = field(default_factory=SegmentationLossConfig)


def run_evaluation(config: EvaluationRunConfig, *, store: FeatureStore) -> dict[str, float]:
    """Load a checkpointed decoder and evaluate it over a dataset.

    Parameters
    ----------
    config : EvaluationRunConfig
    store : FeatureStore
        An already-populated store (see :mod:`~precisionai.agritune.services.feature_service`).

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

    batches = build_training_batches(dataset, batch_size=config.batch_size)

    first_sample = dataset[0]
    probe_sample = PreparedSample(sample_id=first_sample.sample_id, image=None, target=None)
    patch_dim, cls_dim = probe_feature_dims(provider, probe_sample)
    output_size = tuple(np.array(first_sample.target).shape)

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

    task = SegmentationTask(decoder, SegmentationLoss(config.loss, num_classes=config.num_classes))
    metric = SegmentationMetric(num_classes=config.num_classes)
    return evaluate(task, provider, batches, metric)
