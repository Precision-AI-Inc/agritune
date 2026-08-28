# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.schemas.features.EncoderFeatures."""

import pytest
import torch

from precisionai.agritune.schemas.features import EncoderFeatures, concatenate_encoder_features


def _make_features(
    *,
    batch_size: int = 2,
    num_patches: int = 6,
    patch_dim: int = 8,
    cls_dim: int | None = 5,
    patch_grid: list[tuple[int, int]] | None = None,
    valid_patch_mask: torch.Tensor | None = None,
    image_sizes: list[tuple[int, int]] | None = None,
) -> EncoderFeatures:
    grid = patch_grid or [(2, 3)] * batch_size
    return EncoderFeatures(
        patch_tokens=torch.randn(batch_size, num_patches, patch_dim),
        cls_tokens=torch.randn(batch_size, cls_dim) if cls_dim is not None else None,
        patch_grid=torch.tensor(grid, dtype=torch.long),
        valid_patch_mask=valid_patch_mask,
        image_sizes=image_sizes or [(224, 224)] * batch_size,
        encoder_model="pai-embedding",
        encoder_revision=None,
    )


def test_construction_with_cls_tokens() -> None:
    features = _make_features()
    assert features.batch_size == 2
    assert features.cls_tokens is not None
    assert features.cls_tokens.shape == (2, 5)


def test_construction_without_cls_tokens() -> None:
    features = _make_features(cls_dim=None)
    assert features.cls_tokens is None


def test_patch_dim_may_differ_from_cls_dim() -> None:
    features = _make_features(patch_dim=8, cls_dim=5)
    assert features.cls_tokens is not None
    assert features.patch_tokens.shape[-1] != features.cls_tokens.shape[-1]


def test_rectangular_patch_grid() -> None:
    features = _make_features(num_patches=20, patch_grid=[(4, 5), (4, 5)])
    assert features.patch_grid_hw(0) == (4, 5)


def test_variable_image_resolution_across_batch() -> None:
    features = _make_features(image_sizes=[(224, 224), (512, 384)])
    assert features.image_sizes[0] != features.image_sizes[1]


def test_uniform_patch_grid_returns_shared_grid() -> None:
    features = _make_features(patch_grid=[(2, 3), (2, 3)])
    assert features.uniform_patch_grid() == (2, 3)


def test_uniform_patch_grid_raises_on_mismatch() -> None:
    features = _make_features(num_patches=6, patch_grid=[(2, 3), (3, 2)])
    with pytest.raises(ValueError, match="non-uniform patch grids"):
        features.uniform_patch_grid()


def test_variable_patch_count_via_valid_patch_mask() -> None:
    mask = torch.tensor([[True, True, True, True, True, True], [True, True, True, True, False, False]])
    features = _make_features(num_patches=6, patch_grid=[(2, 3), (2, 2)], valid_patch_mask=mask)
    assert features.patch_grid_hw(1) == (2, 2)


def test_patch_tokens_wrong_ndim_raises() -> None:
    with pytest.raises(ValueError, match="patch_tokens must have shape"):
        EncoderFeatures(
            patch_tokens=torch.randn(2, 8),
            cls_tokens=None,
            patch_grid=torch.tensor([[2, 4], [2, 4]]),
            valid_patch_mask=None,
            image_sizes=[(224, 224), (224, 224)],
            encoder_model="pai-embedding",
            encoder_revision=None,
        )


def test_cls_tokens_batch_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="cls_tokens must have shape"):
        EncoderFeatures(
            patch_tokens=torch.randn(2, 6, 8),
            cls_tokens=torch.randn(3, 5),
            patch_grid=torch.tensor([[2, 3], [2, 3]]),
            valid_patch_mask=None,
            image_sizes=[(224, 224), (224, 224)],
            encoder_model="pai-embedding",
            encoder_revision=None,
        )


def test_patch_grid_wrong_shape_raises() -> None:
    with pytest.raises(ValueError, match="patch_grid must have shape"):
        EncoderFeatures(
            patch_tokens=torch.randn(2, 6, 8),
            cls_tokens=None,
            patch_grid=torch.tensor([1, 2, 3]),
            valid_patch_mask=None,
            image_sizes=[(224, 224), (224, 224)],
            encoder_model="pai-embedding",
            encoder_revision=None,
        )


def test_valid_patch_mask_wrong_shape_raises() -> None:
    with pytest.raises(ValueError, match="valid_patch_mask must have shape"):
        EncoderFeatures(
            patch_tokens=torch.randn(2, 6, 8),
            cls_tokens=None,
            patch_grid=torch.tensor([[2, 3], [2, 3]]),
            valid_patch_mask=torch.ones(2, 4, dtype=torch.bool),
            image_sizes=[(224, 224), (224, 224)],
            encoder_model="pai-embedding",
            encoder_revision=None,
        )


def test_image_sizes_wrong_length_raises() -> None:
    with pytest.raises(ValueError, match="image_sizes must have length"):
        EncoderFeatures(
            patch_tokens=torch.randn(2, 6, 8),
            cls_tokens=None,
            patch_grid=torch.tensor([[2, 3], [2, 3]]),
            valid_patch_mask=None,
            image_sizes=[(224, 224)],
            encoder_model="pai-embedding",
            encoder_revision=None,
        )


def test_patch_grid_product_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="patches but"):
        EncoderFeatures(
            patch_tokens=torch.randn(1, 6, 8),
            cls_tokens=None,
            patch_grid=torch.tensor([[3, 3]]),
            valid_patch_mask=None,
            image_sizes=[(224, 224)],
            encoder_model="pai-embedding",
            encoder_revision=None,
        )


def test_metadata_defaults_to_empty_dict() -> None:
    features = _make_features()
    assert features.metadata == {}


def test_concatenate_combines_batch_dimension() -> None:
    first = _make_features(batch_size=2, num_patches=6, patch_grid=[(2, 3), (2, 3)])
    second = _make_features(batch_size=1, num_patches=6, patch_grid=[(2, 3)])
    combined = concatenate_encoder_features([first, second])
    assert combined.batch_size == 3
    assert combined.patch_tokens.shape == (3, 6, 8)


def test_concatenate_pads_to_largest_patch_count() -> None:
    small = _make_features(batch_size=1, num_patches=4, patch_grid=[(2, 2)])
    large = _make_features(batch_size=1, num_patches=6, patch_grid=[(2, 3)])
    combined = concatenate_encoder_features([small, large])
    assert combined.patch_tokens.shape == (2, 6, 8)
    assert combined.valid_patch_mask is not None
    assert combined.valid_patch_mask[0].sum().item() == 4
    assert combined.valid_patch_mask[1].sum().item() == 6


def test_concatenate_preserves_an_entrys_own_valid_patch_mask() -> None:
    own_mask = torch.tensor([[True, True, True, False]])
    padded_entry = _make_features(batch_size=1, num_patches=4, patch_grid=[(1, 3)], valid_patch_mask=own_mask)
    larger_entry = _make_features(batch_size=1, num_patches=6, patch_grid=[(2, 3)])

    combined = concatenate_encoder_features([padded_entry, larger_entry])

    assert combined.valid_patch_mask is not None
    assert combined.valid_patch_mask[0].tolist() == [True, True, True, False, False, False]


def test_concatenate_single_entry_is_a_no_op() -> None:
    only = _make_features(batch_size=2)
    combined = concatenate_encoder_features([only])
    assert combined.batch_size == 2
    assert combined.valid_patch_mask is None


def test_concatenate_preserves_cls_tokens_when_all_present() -> None:
    first = _make_features(batch_size=1, cls_dim=5)
    second = _make_features(batch_size=1, cls_dim=5)
    combined = concatenate_encoder_features([first, second])
    assert combined.cls_tokens is not None
    assert combined.cls_tokens.shape == (2, 5)


def test_concatenate_empty_list_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        concatenate_encoder_features([])


def test_concatenate_rejects_mismatched_patch_dim() -> None:
    first = _make_features(batch_size=1, patch_dim=8)
    second = _make_features(batch_size=1, patch_dim=4)
    with pytest.raises(ValueError, match="patch dimension"):
        concatenate_encoder_features([first, second])


def test_concatenate_rejects_mismatched_cls_presence() -> None:
    first = _make_features(batch_size=1, cls_dim=5)
    second = _make_features(batch_size=1, cls_dim=None)
    with pytest.raises(ValueError, match="cls_tokens is present"):
        concatenate_encoder_features([first, second])


def test_concatenate_rejects_mismatched_encoder_identity() -> None:
    first = _make_features(batch_size=1)
    second = EncoderFeatures(
        patch_tokens=first.patch_tokens.clone(),
        cls_tokens=first.cls_tokens.clone() if first.cls_tokens is not None else None,
        patch_grid=first.patch_grid.clone(),
        valid_patch_mask=None,
        image_sizes=list(first.image_sizes),
        encoder_model="different-model",
        encoder_revision=None,
    )
    with pytest.raises(ValueError, match="encoder_model/encoder_revision"):
        concatenate_encoder_features([first, second])
