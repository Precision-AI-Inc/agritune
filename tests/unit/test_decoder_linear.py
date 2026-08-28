# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.tasks.segmentation.decoders.linear.LinearProbeDecoder."""

import pytest
import torch

from precisionai.agritune.schemas.features import EncoderFeatures
from precisionai.agritune.tasks.segmentation.decoders.linear import LinearProbeDecoder


def _features(*, batch_size: int = 2, grid: tuple[int, int] = (4, 4), patch_dim: int = 8) -> EncoderFeatures:
    height, width = grid
    return EncoderFeatures(
        patch_tokens=torch.randn(batch_size, height * width, patch_dim),
        cls_tokens=None,
        patch_grid=torch.tensor([[height, width]] * batch_size),
        valid_patch_mask=None,
        image_sizes=[(224, 224)] * batch_size,
        encoder_model="fake",
        encoder_revision=None,
    )


def test_output_shape_matches_output_size_and_num_classes() -> None:
    decoder = LinearProbeDecoder(patch_dim=8, num_classes=5, output_size=(64, 64))
    logits = decoder(_features(batch_size=3, grid=(4, 4), patch_dim=8))
    assert logits.shape == (3, 5, 64, 64)


def test_non_uniform_patch_grid_raises() -> None:
    decoder = LinearProbeDecoder(patch_dim=8, num_classes=5, output_size=(32, 32))
    features = EncoderFeatures(
        patch_tokens=torch.randn(2, 16, 8),
        cls_tokens=None,
        patch_grid=torch.tensor([[4, 4], [2, 8]]),
        valid_patch_mask=None,
        image_sizes=[(224, 224), (224, 224)],
        encoder_model="fake",
        encoder_revision=None,
    )
    with pytest.raises(ValueError, match="non-uniform patch grids"):
        decoder(features)


def test_gradients_flow_to_projection_weight() -> None:
    decoder = LinearProbeDecoder(patch_dim=8, num_classes=3, output_size=(16, 16))
    logits = decoder(_features(grid=(2, 2), patch_dim=8))
    logits.sum().backward()
    assert decoder.projection.weight.grad is not None
    assert torch.any(decoder.projection.weight.grad != 0)


def test_rectangular_output_size() -> None:
    decoder = LinearProbeDecoder(patch_dim=8, num_classes=2, output_size=(30, 50))
    logits = decoder(_features(grid=(3, 5), patch_dim=8))
    assert logits.shape == (2, 2, 30, 50)
