# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.tasks.segmentation.decoders.token_fpn.TokenFPNDecoder."""

import pytest
import torch

from precisionai.agritune.schemas.features import EncoderFeatures
from precisionai.agritune.tasks.segmentation.decoders.token_fpn import CLSFusion, TokenFPNDecoder


def _features(
    *, batch_size: int = 2, grid: tuple[int, int] = (4, 4), patch_dim: int = 8, cls_dim: int | None = None
) -> EncoderFeatures:
    height, width = grid
    return EncoderFeatures(
        patch_tokens=torch.randn(batch_size, height * width, patch_dim),
        cls_tokens=torch.randn(batch_size, cls_dim) if cls_dim is not None else None,
        patch_grid=torch.tensor([[height, width]] * batch_size),
        valid_patch_mask=None,
        image_sizes=[(224, 224)] * batch_size,
        encoder_model="fake",
        encoder_revision=None,
    )


def test_output_shape_with_no_cls_fusion() -> None:
    decoder = TokenFPNDecoder(patch_dim=8, num_classes=4, output_size=(64, 64))
    logits = decoder(_features(batch_size=2, grid=(4, 4), patch_dim=8))
    assert logits.shape == (2, 4, 64, 64)


def test_output_shape_with_more_refinement_layers() -> None:
    decoder = TokenFPNDecoder(patch_dim=8, num_classes=4, output_size=(64, 64), num_layers=4)
    assert len(decoder.refine) == 4 * 2  # each layer is a (Conv2d, ReLU) pair
    logits = decoder(_features(batch_size=2, grid=(4, 4), patch_dim=8))
    assert logits.shape == (2, 4, 64, 64)


def test_num_layers_must_be_positive() -> None:
    with pytest.raises(ValueError, match="num_layers must be positive"):
        TokenFPNDecoder(patch_dim=8, num_classes=4, output_size=(32, 32), num_layers=0)


def test_cls_fusion_requires_cls_dim() -> None:
    with pytest.raises(ValueError, match="cls_fusion='concat' requires cls_dim"):
        TokenFPNDecoder(patch_dim=8, num_classes=4, output_size=(32, 32), cls_fusion=CLSFusion.CONCAT)


def test_concat_cls_fusion_produces_correct_shape() -> None:
    decoder = TokenFPNDecoder(patch_dim=8, num_classes=4, output_size=(32, 32), cls_dim=6, cls_fusion=CLSFusion.CONCAT)
    logits = decoder(_features(grid=(4, 4), patch_dim=8, cls_dim=6))
    assert logits.shape == (2, 4, 32, 32)


def test_film_cls_fusion_produces_correct_shape() -> None:
    decoder = TokenFPNDecoder(patch_dim=8, num_classes=4, output_size=(32, 32), cls_dim=6, cls_fusion=CLSFusion.FILM)
    logits = decoder(_features(grid=(4, 4), patch_dim=8, cls_dim=6))
    assert logits.shape == (2, 4, 32, 32)


def test_concat_fusion_without_cls_tokens_in_features_raises() -> None:
    decoder = TokenFPNDecoder(patch_dim=8, num_classes=4, output_size=(32, 32), cls_dim=6, cls_fusion=CLSFusion.CONCAT)
    with pytest.raises(ValueError, match="requires cls_tokens"):
        decoder(_features(grid=(4, 4), patch_dim=8, cls_dim=None))


def test_film_fusion_without_cls_tokens_in_features_raises() -> None:
    decoder = TokenFPNDecoder(patch_dim=8, num_classes=4, output_size=(32, 32), cls_dim=6, cls_fusion=CLSFusion.FILM)
    with pytest.raises(ValueError, match="requires cls_tokens"):
        decoder(_features(grid=(4, 4), patch_dim=8, cls_dim=None))


def test_non_uniform_patch_grid_raises() -> None:
    decoder = TokenFPNDecoder(patch_dim=8, num_classes=4, output_size=(32, 32))
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


def test_gradients_flow_through_refinement_stack() -> None:
    decoder = TokenFPNDecoder(patch_dim=8, num_classes=3, output_size=(16, 16))
    logits = decoder(_features(grid=(2, 2), patch_dim=8))
    logits.sum().backward()
    assert decoder.channel_proj.weight.grad is not None
    assert torch.any(decoder.channel_proj.weight.grad != 0)


def test_cls_fusion_none_ignores_cls_tokens_even_if_present() -> None:
    decoder = TokenFPNDecoder(patch_dim=8, num_classes=4, output_size=(16, 16))
    features = _features(grid=(2, 2), patch_dim=8, cls_dim=6)  # cls_tokens present but unused
    logits = decoder(features)
    assert logits.shape == (2, 4, 16, 16)
