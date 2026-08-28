# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0
"""Feature-space augmentation (patch dropout, token masking, ...) — independent of image augmentation."""

from precisionai.agritune.augmentations.feature.pipeline import FeatureAugmentationConfig, FeatureAugmentationPipeline
from precisionai.agritune.augmentations.feature.transforms import (
    cls_dropout,
    feature_channel_dropout,
    gaussian_feature_noise,
    patch_dropout,
    token_masking,
)

__all__ = [
    "FeatureAugmentationConfig",
    "FeatureAugmentationPipeline",
    "cls_dropout",
    "feature_channel_dropout",
    "gaussian_feature_noise",
    "patch_dropout",
    "token_masking",
]
