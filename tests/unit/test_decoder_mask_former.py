# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.tasks.segmentation.decoders.mask_former.MaskFormerDecoder."""

import pytest
import torch

from precisionai.agritune.schemas.features import EncoderFeatures
from precisionai.agritune.tasks.segmentation.decoders.mask_former import MaskFormerDecoder


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


def _decoder(**overrides: object) -> MaskFormerDecoder:
    defaults: dict[str, object] = {
        "patch_dim": 8,
        "num_classes": 5,
        "output_size": (16, 16),
        "hidden_dim": 8,
        "num_queries": 6,
        "num_layers": 1,
        "num_heads": 2,
    }
    defaults.update(overrides)
    return MaskFormerDecoder(**defaults)  # type: ignore[arg-type]


def test_output_shape_matches_output_size_and_num_classes() -> None:
    decoder = _decoder(output_size=(64, 64))
    logits = decoder(_features(batch_size=3, grid=(4, 4), patch_dim=8))
    assert logits.shape == (3, 5, 64, 64)


def test_hidden_dim_not_divisible_by_num_heads_raises() -> None:
    with pytest.raises(ValueError, match="divisible"):
        _decoder(hidden_dim=8, num_heads=3)


def test_num_queries_is_independent_of_num_classes() -> None:
    decoder = _decoder(num_classes=5, num_queries=11, hidden_dim=8)
    assert decoder.queries.shape == (11, 8)
    assert decoder.class_head.out_features == 6  # num_classes + 1 "no object" class


def test_output_values_are_non_negative() -> None:
    # combined = sum_q class_probs(q, c) * sigmoid(mask_logits(q)) — a sum of products of two
    # non-negative terms (a softmax output and a sigmoid output), so it can never go negative.
    decoder = _decoder(output_size=(8, 8))
    logits = decoder(_features(grid=(2, 2), patch_dim=8))
    assert torch.all(logits >= 0.0)


def test_non_uniform_patch_grid_raises() -> None:
    decoder = _decoder()
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


def test_gradients_flow_to_queries() -> None:
    decoder = _decoder(output_size=(16, 16))
    logits = decoder(_features(grid=(2, 2), patch_dim=8))
    logits.sum().backward()
    assert decoder.queries.grad is not None
    assert torch.any(decoder.queries.grad != 0)


def test_rectangular_output_size() -> None:
    decoder = _decoder(num_classes=2, output_size=(30, 50))
    logits = decoder(_features(grid=(3, 5), patch_dim=8))
    assert logits.shape == (2, 2, 30, 50)
