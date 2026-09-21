# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.tasks.segmentation.decoders.pyramid_pooling.PyramidPoolingDecoder."""

import pytest
import torch

from precisionai.agritune.schemas.features import EncoderFeatures
from precisionai.agritune.tasks.segmentation.decoders.pyramid_pooling import PyramidPoolingDecoder


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
    decoder = PyramidPoolingDecoder(patch_dim=8, num_classes=5, output_size=(64, 64), hidden_dim=16)
    logits = decoder(_features(batch_size=3, grid=(4, 4), patch_dim=8))
    assert logits.shape == (3, 5, 64, 64)


def test_output_shape_with_more_fuse_layers() -> None:
    decoder = PyramidPoolingDecoder(patch_dim=8, num_classes=5, output_size=(64, 64), hidden_dim=16, num_layers=3)
    assert len(decoder.fuse) == 3 * 2  # each layer is a (Conv2d, ReLU) pair
    logits = decoder(_features(batch_size=3, grid=(4, 4), patch_dim=8))
    assert logits.shape == (3, 5, 64, 64)


def test_num_layers_must_be_positive() -> None:
    with pytest.raises(ValueError, match="num_layers must be positive"):
        PyramidPoolingDecoder(patch_dim=8, num_classes=4, output_size=(32, 32), num_layers=0)


def test_custom_pool_sizes_are_stored_and_produce_one_pool_each() -> None:
    decoder = PyramidPoolingDecoder(patch_dim=8, num_classes=4, output_size=(8, 8), hidden_dim=8, pool_sizes=(1, 2))
    assert decoder.pool_sizes == (1, 2)
    assert len(decoder.pools) == 2


def test_tiny_patch_grid_does_not_raise() -> None:
    decoder = PyramidPoolingDecoder(patch_dim=8, num_classes=2, output_size=(4, 4), hidden_dim=8)
    logits = decoder(_features(grid=(1, 1), patch_dim=8))
    assert logits.shape == (2, 2, 4, 4)


def test_non_uniform_patch_grid_raises() -> None:
    decoder = PyramidPoolingDecoder(patch_dim=8, num_classes=5, output_size=(32, 32))
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


def test_gradients_flow_to_classifier_weight() -> None:
    decoder = PyramidPoolingDecoder(patch_dim=8, num_classes=3, output_size=(16, 16), hidden_dim=8)
    logits = decoder(_features(grid=(2, 2), patch_dim=8))
    logits.sum().backward()
    assert decoder.classifier.weight.grad is not None
    assert torch.any(decoder.classifier.weight.grad != 0)


def test_rectangular_output_size() -> None:
    decoder = PyramidPoolingDecoder(patch_dim=8, num_classes=2, output_size=(30, 50), hidden_dim=8)
    logits = decoder(_features(grid=(3, 5), patch_dim=8))
    assert logits.shape == (2, 2, 30, 50)
