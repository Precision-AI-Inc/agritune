# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0
"""Segmentation task: losses, metrics, postprocessing, and decoders."""

from precisionai.agritune.tasks.segmentation.losses import (
    DiceLoss,
    SegmentationLoss,
    SegmentationLossConfig,
    bce_with_logits_loss,
    cross_entropy_loss,
)
from precisionai.agritune.tasks.segmentation.metrics import SegmentationMetric
from precisionai.agritune.tasks.segmentation.postprocessing import logits_to_predictions, resize_predictions
from precisionai.agritune.tasks.segmentation.task import SegmentationTask

__all__ = [
    "DiceLoss",
    "SegmentationLoss",
    "SegmentationLossConfig",
    "SegmentationMetric",
    "SegmentationTask",
    "bce_with_logits_loss",
    "cross_entropy_loss",
    "logits_to_predictions",
    "resize_predictions",
]
