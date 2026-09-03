# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.augmentations.feature.pipeline.FeatureAugmentationPipeline."""

import torch

from precisionai.agritune.augmentations.feature.pipeline import FeatureAugmentationConfig, FeatureAugmentationPipeline
from precisionai.agritune.schemas.features import EncoderFeatures
from precisionai.agritune.schemas.protocols import FeatureAugmentation


def _make_features(*, batch_size: int = 4, num_patches: int = 6, patch_dim: int = 8) -> EncoderFeatures:
    return EncoderFeatures(
        patch_tokens=torch.ones(batch_size, num_patches, patch_dim),
        cls_tokens=torch.ones(batch_size, 5),
        patch_grid=torch.tensor([[2, 3]] * batch_size, dtype=torch.long),
        valid_patch_mask=None,
        image_sizes=[(224, 224)] * batch_size,
        encoder_model="fake-encoder",
        encoder_revision=None,
    )


def test_default_config_is_a_no_op() -> None:
    features = _make_features()
    pipeline = FeatureAugmentationPipeline()
    result = pipeline.apply(features, generator=torch.Generator().manual_seed(0))
    assert result.cls_tokens is not None
    assert features.cls_tokens is not None
    assert torch.equal(result.patch_tokens, features.patch_tokens)
    assert torch.equal(result.cls_tokens, features.cls_tokens)


def test_enabled_transforms_change_the_output() -> None:
    features = _make_features()
    pipeline = FeatureAugmentationPipeline(
        FeatureAugmentationConfig(patch_dropout_probability=0.5, gaussian_noise_std=0.1)
    )
    result = pipeline.apply(features, generator=torch.Generator().manual_seed(0))
    assert not torch.equal(result.patch_tokens, features.patch_tokens)


def test_pipeline_is_deterministic_for_the_same_generator_seed() -> None:
    features = _make_features()
    pipeline = FeatureAugmentationPipeline(
        FeatureAugmentationConfig(
            patch_dropout_probability=0.3,
            token_masking_probability=0.2,
            gaussian_noise_std=0.05,
            cls_dropout_probability=0.4,
            channel_dropout_probability=0.3,
        )
    )
    first = pipeline.apply(features, generator=torch.Generator().manual_seed(42))
    second = pipeline.apply(features, generator=torch.Generator().manual_seed(42))
    assert torch.equal(first.patch_tokens, second.patch_tokens)
    assert first.cls_tokens is not None
    assert second.cls_tokens is not None
    assert torch.equal(first.cls_tokens, second.cls_tokens)


def test_pipeline_uses_the_global_rng_when_no_generator_given() -> None:
    features = _make_features()
    pipeline = FeatureAugmentationPipeline(FeatureAugmentationConfig(gaussian_noise_std=0.1))
    torch.manual_seed(0)
    result = pipeline.apply(features)
    assert not torch.equal(result.patch_tokens, features.patch_tokens)


def test_feature_augmentation_pipeline_satisfies_protocol() -> None:
    assert isinstance(FeatureAugmentationPipeline(), FeatureAugmentation)
