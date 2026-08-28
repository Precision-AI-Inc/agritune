# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0
"""Core dataclasses and protocols: EncoderFeatures, Sample, PreparedSample, and interfaces."""

from precisionai.agritune.schemas.augmentation import AugmentationRecord, TransformRecord
from precisionai.agritune.schemas.features import EncoderFeatures, concatenate_encoder_features
from precisionai.agritune.schemas.protocols import (
    Decoder,
    EncoderBackend,
    FeatureProvider,
    FeatureStore,
    Metric,
    Task,
    Tracker,
)
from precisionai.agritune.schemas.samples import PreparedSample, Sample

__all__ = [
    "AugmentationRecord",
    "Decoder",
    "EncoderBackend",
    "EncoderFeatures",
    "FeatureProvider",
    "FeatureStore",
    "Metric",
    "PreparedSample",
    "Sample",
    "Task",
    "Tracker",
    "TransformRecord",
    "concatenate_encoder_features",
]
