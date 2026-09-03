# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.augmentations.feature.transforms."""

import pytest
import torch

from precisionai.agritune.augmentations.feature.transforms import (
    cls_dropout,
    feature_channel_dropout,
    gaussian_feature_noise,
    patch_dropout,
    token_masking,
)
from precisionai.agritune.schemas.features import EncoderFeatures


def _make_features(
    *, batch_size: int = 4, num_patches: int = 6, patch_dim: int = 8, cls_dim: int | None = 5
) -> EncoderFeatures:
    return EncoderFeatures(
        patch_tokens=torch.ones(batch_size, num_patches, patch_dim),
        cls_tokens=torch.ones(batch_size, cls_dim) if cls_dim is not None else None,
        patch_grid=torch.tensor([[2, 3]] * batch_size, dtype=torch.long),
        valid_patch_mask=None,
        image_sizes=[(224, 224)] * batch_size,
        encoder_model="fake-encoder",
        encoder_revision=None,
    )


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_patch_dropout_zeros_some_patches_and_rescales_survivors() -> None:
    features = _make_features()
    result = patch_dropout(features, probability=0.5, generator=_generator(0))

    per_patch_values = result.patch_tokens[:, :, 0]  # every channel is identical (all-ones input)
    zeroed = (per_patch_values == 0).sum().item()
    survivors = (per_patch_values != 0).sum().item()
    assert zeroed > 0
    assert survivors > 0
    # surviving patches are rescaled by 1 / (1 - 0.5) = 2.0
    assert torch.allclose(per_patch_values[per_patch_values != 0], torch.tensor(2.0))


def test_patch_dropout_zero_probability_is_a_no_op() -> None:
    features = _make_features()
    result = patch_dropout(features, probability=0.0, generator=_generator(0))
    assert torch.equal(result.patch_tokens, features.patch_tokens)


def test_patch_dropout_rejects_probability_of_one() -> None:
    features = _make_features()
    with pytest.raises(ValueError, match=r"probability must be in \[0, 1\)"):
        patch_dropout(features, probability=1.0, generator=_generator(0))


def test_patch_dropout_is_deterministic_for_the_same_seed() -> None:
    features = _make_features()
    first = patch_dropout(features, probability=0.5, generator=_generator(7))
    second = patch_dropout(features, probability=0.5, generator=_generator(7))
    assert torch.equal(first.patch_tokens, second.patch_tokens)


def test_token_masking_sets_masked_patches_to_the_mask_value_without_rescaling() -> None:
    features = _make_features()
    result = token_masking(features, probability=0.5, mask_value=-1.0, generator=_generator(0))

    values = result.patch_tokens[:, :, 0]
    masked = (values == -1.0).sum().item()
    unmasked = (values == 1.0).sum().item()  # unaugmented survivors keep their original value (no rescale)
    assert masked > 0
    assert unmasked > 0


def test_token_masking_allows_probability_of_one() -> None:
    features = _make_features()
    result = token_masking(features, probability=1.0, mask_value=0.0, generator=_generator(0))
    assert torch.all(result.patch_tokens == 0.0)


def test_token_masking_zero_probability_is_a_no_op() -> None:
    features = _make_features()
    result = token_masking(features, probability=0.0, generator=_generator(0))
    assert torch.equal(result.patch_tokens, features.patch_tokens)


def test_token_masking_invalid_probability_raises() -> None:
    features = _make_features()
    with pytest.raises(ValueError, match=r"probability must be in \[0, 1\]"):
        token_masking(features, probability=1.5, generator=_generator(0))


def test_gaussian_feature_noise_perturbs_patch_and_cls_tokens() -> None:
    features = _make_features()
    result = gaussian_feature_noise(features, std=0.1, generator=_generator(0))
    assert not torch.equal(result.patch_tokens, features.patch_tokens)
    assert result.cls_tokens is not None
    assert features.cls_tokens is not None
    assert not torch.equal(result.cls_tokens, features.cls_tokens)


def test_gaussian_feature_noise_zero_std_is_a_no_op() -> None:
    features = _make_features()
    result = gaussian_feature_noise(features, std=0.0, generator=_generator(0))
    assert torch.equal(result.patch_tokens, features.patch_tokens)


def test_gaussian_feature_noise_negative_std_raises() -> None:
    features = _make_features()
    with pytest.raises(ValueError, match="std must be >= 0"):
        gaussian_feature_noise(features, std=-0.1, generator=_generator(0))


def test_gaussian_feature_noise_handles_missing_cls_tokens() -> None:
    features = _make_features(cls_dim=None)
    result = gaussian_feature_noise(features, std=0.1, generator=_generator(0))
    assert result.cls_tokens is None


def test_cls_dropout_zeros_some_samples_cls_and_rescales_survivors() -> None:
    features = _make_features(batch_size=20)
    result = cls_dropout(features, probability=0.5, generator=_generator(0))
    assert result.cls_tokens is not None

    per_sample = result.cls_tokens[:, 0]
    zeroed = (per_sample == 0).sum().item()
    survivors = (per_sample != 0).sum().item()
    assert zeroed > 0
    assert survivors > 0
    assert torch.allclose(per_sample[per_sample != 0], torch.tensor(2.0))


def test_cls_dropout_is_a_no_op_when_cls_tokens_absent() -> None:
    features = _make_features(cls_dim=None)
    result = cls_dropout(features, probability=0.5, generator=_generator(0))
    assert result.cls_tokens is None


def test_cls_dropout_rejects_probability_of_one() -> None:
    features = _make_features()
    with pytest.raises(ValueError, match=r"probability must be in \[0, 1\)"):
        cls_dropout(features, probability=1.0, generator=_generator(0))


def test_feature_channel_dropout_zeros_whole_channels_across_the_batch() -> None:
    features = _make_features(patch_dim=32, cls_dim=32)
    result = feature_channel_dropout(features, probability=0.5, generator=_generator(0))

    # a dropped channel is zero for every patch and every sample; a surviving one is rescaled
    # identically everywhere — so each channel, viewed across (batch, patch), is uniform.
    per_channel = result.patch_tokens.reshape(-1, result.patch_tokens.shape[-1])
    zeroed_channels = (per_channel == 0).all(dim=0).sum().item()
    surviving_channels = (per_channel != 0).all(dim=0).sum().item()
    assert zeroed_channels + surviving_channels == 32
    assert zeroed_channels > 0
    assert surviving_channels > 0


def test_feature_channel_dropout_without_cls_tokens_only_masks_patches() -> None:
    features = _make_features(cls_dim=None, patch_dim=16)
    result = feature_channel_dropout(features, probability=0.5, generator=_generator(0))
    assert result.cls_tokens is None
    assert result.patch_tokens.shape == features.patch_tokens.shape


def test_feature_channel_dropout_uses_independent_masks_for_patch_and_cls_dims() -> None:
    features = _make_features(patch_dim=4, cls_dim=6)
    result = feature_channel_dropout(features, probability=0.5, generator=_generator(0))
    assert result.patch_tokens.shape[-1] == 4
    assert result.cls_tokens is not None
    assert result.cls_tokens.shape[-1] == 6


def test_feature_channel_dropout_zero_probability_is_a_no_op() -> None:
    features = _make_features()
    result = feature_channel_dropout(features, probability=0.0, generator=_generator(0))
    assert torch.equal(result.patch_tokens, features.patch_tokens)


def test_feature_channel_dropout_rejects_probability_of_one() -> None:
    features = _make_features()
    with pytest.raises(ValueError, match=r"probability must be in \[0, 1\)"):
        feature_channel_dropout(features, probability=1.0, generator=_generator(0))
