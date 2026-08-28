# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.tasks.segmentation.decoders.mlp_probe.MLPProbeDecoder."""

import pytest
import torch

from precisionai.agritune.schemas.features import EncoderFeatures
from precisionai.agritune.tasks.segmentation.decoders.mlp_probe import MLPProbeDecoder


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
    decoder = MLPProbeDecoder(patch_dim=8, num_classes=5, output_size=(64, 64))
    logits = decoder(_features(batch_size=3, grid=(4, 4), patch_dim=8))
    assert logits.shape == (3, 5, 64, 64)


def test_default_hidden_dims_is_a_single_linear_layer() -> None:
    decoder = MLPProbeDecoder(patch_dim=8, num_classes=5, output_size=(16, 16))
    assert decoder.hidden_dims == ()
    assert len(decoder.mlp) == 1
    assert isinstance(decoder.mlp[0], torch.nn.Linear)
    assert decoder.mlp[0].in_features == 8
    assert decoder.mlp[0].out_features == 5


def test_hidden_dims_adds_intermediate_layers() -> None:
    decoder = MLPProbeDecoder(patch_dim=8, num_classes=5, output_size=(16, 16), hidden_dims=(32, 16))
    logits = decoder(_features(grid=(4, 4), patch_dim=8))
    assert logits.shape == (2, 5, 16, 16)

    linear_layers = [layer for layer in decoder.mlp if isinstance(layer, torch.nn.Linear)]
    assert [layer.out_features for layer in linear_layers] == [32, 16, 5]


def test_dropout_layers_inserted_after_each_hidden_activation() -> None:
    decoder = MLPProbeDecoder(patch_dim=8, num_classes=3, output_size=(8, 8), hidden_dims=(16,), dropout=0.5)
    assert any(isinstance(layer, torch.nn.Dropout) for layer in decoder.mlp)


def test_non_uniform_patch_grid_raises() -> None:
    decoder = MLPProbeDecoder(patch_dim=8, num_classes=5, output_size=(32, 32))
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


def test_gradients_flow_to_every_linear_layer() -> None:
    decoder = MLPProbeDecoder(patch_dim=8, num_classes=3, output_size=(16, 16), hidden_dims=(16,))
    logits = decoder(_features(grid=(2, 2), patch_dim=8))
    logits.sum().backward()
    for layer in decoder.mlp:
        if isinstance(layer, torch.nn.Linear):
            assert layer.weight.grad is not None
            assert torch.any(layer.weight.grad != 0)


def test_rectangular_output_size() -> None:
    decoder = MLPProbeDecoder(patch_dim=8, num_classes=2, output_size=(30, 50))
    logits = decoder(_features(grid=(3, 5), patch_dim=8))
    assert logits.shape == (2, 2, 30, 50)
