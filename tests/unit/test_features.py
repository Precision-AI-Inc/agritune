# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.schemas.features.EncoderFeatures."""

from dataclasses import replace

import pytest
import torch

from precisionai.agritune.schemas.features import EncoderFeatures, concatenate_encoder_features, select_one


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


def test_patch_tokens_reject_nonpositive_dimensions() -> None:
    with pytest.raises(ValueError, match="patch_tokens dimensions B, N, and D must be positive"):
        EncoderFeatures(
            patch_tokens=torch.ones(1, 0, 3),
            cls_tokens=None,
            patch_grid=torch.tensor([[1, 1]]),
            valid_patch_mask=None,
            image_sizes=[(8, 8)],
            encoder_model="pai-embedding",
            encoder_revision=None,
        )


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


def test_patch_tokens_reject_integer_dtype() -> None:
    with pytest.raises(ValueError, match="patch_tokens must use a floating-point dtype"):
        EncoderFeatures(
            patch_tokens=torch.ones(1, 2, 3, dtype=torch.int64),
            cls_tokens=None,
            patch_grid=torch.tensor([[1, 2]]),
            valid_patch_mask=None,
            image_sizes=[(8, 8)],
            encoder_model="pai-embedding",
            encoder_revision=None,
        )


def test_patch_tokens_reject_nonfinite_values() -> None:
    tokens = torch.ones(1, 2, 3)
    tokens[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="patch_tokens must contain only finite values"):
        EncoderFeatures(
            patch_tokens=tokens,
            cls_tokens=None,
            patch_grid=torch.tensor([[1, 2]]),
            valid_patch_mask=None,
            image_sizes=[(8, 8)],
            encoder_model="pai-embedding",
            encoder_revision=None,
        )


def test_cls_tokens_reject_integer_dtype() -> None:
    with pytest.raises(ValueError, match="cls_tokens must use a floating-point dtype"):
        EncoderFeatures(
            patch_tokens=torch.ones(1, 2, 3),
            cls_tokens=torch.ones(1, 4, dtype=torch.int64),
            patch_grid=torch.tensor([[1, 2]]),
            valid_patch_mask=None,
            image_sizes=[(8, 8)],
            encoder_model="pai-embedding",
            encoder_revision=None,
        )


def test_cls_tokens_reject_zero_class_dimension() -> None:
    with pytest.raises(ValueError, match="cls_tokens must have shape"):
        EncoderFeatures(
            patch_tokens=torch.ones(1, 2, 3),
            cls_tokens=torch.ones(1, 0),
            patch_grid=torch.tensor([[1, 2]]),
            valid_patch_mask=None,
            image_sizes=[(8, 8)],
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


def test_cls_tokens_reject_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="cls_tokens must contain only finite values"):
        EncoderFeatures(
            patch_tokens=torch.ones(1, 2, 3),
            cls_tokens=torch.tensor([[float("inf")]]),
            patch_grid=torch.tensor([[1, 2]]),
            valid_patch_mask=None,
            image_sizes=[(8, 8)],
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


def test_valid_patch_mask_requires_boolean_dtype() -> None:
    with pytest.raises(ValueError, match=r"valid_patch_mask must use dtype torch\.bool"):
        EncoderFeatures(
            patch_tokens=torch.randn(1, 2, 8),
            cls_tokens=None,
            patch_grid=torch.tensor([[1, 2]]),
            valid_patch_mask=torch.ones(1, 2, dtype=torch.int64),
            image_sizes=[(224, 224)],
            encoder_model="pai-embedding",
            encoder_revision=None,
        )


def test_patch_grid_requires_integer_dtype() -> None:
    with pytest.raises(ValueError, match="patch_grid must use an integer dtype"):
        EncoderFeatures(
            patch_tokens=torch.randn(1, 2, 8),
            cls_tokens=None,
            patch_grid=torch.tensor([[1.0, 2.0]]),
            valid_patch_mask=None,
            image_sizes=[(224, 224)],
            encoder_model="pai-embedding",
            encoder_revision=None,
        )


def test_patch_grid_requires_positive_dimensions() -> None:
    with pytest.raises(ValueError, match="patch_grid dimensions must be positive"):
        EncoderFeatures(
            patch_tokens=torch.randn(1, 2, 8),
            cls_tokens=None,
            patch_grid=torch.tensor([[0, 2]]),
            valid_patch_mask=None,
            image_sizes=[(224, 224)],
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


def test_image_sizes_reject_nonpositive_dimensions() -> None:
    with pytest.raises(ValueError, match="must contain two positive integers"):
        EncoderFeatures(
            patch_tokens=torch.randn(1, 2, 3),
            cls_tokens=None,
            patch_grid=torch.tensor([[1, 2]]),
            valid_patch_mask=None,
            image_sizes=[(0, 8)],
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


def test_concatenate_preserves_masks_when_tensor_lengths_are_equal() -> None:
    mask = torch.tensor([[True, False, True, False]])
    first = _make_features(batch_size=1, num_patches=4, patch_grid=[(1, 2)], valid_patch_mask=mask)
    second = _make_features(batch_size=1, num_patches=4, patch_grid=[(1, 2)], valid_patch_mask=mask)

    combined = concatenate_encoder_features([first, second])

    assert combined.valid_patch_mask is not None
    assert torch.equal(combined.valid_patch_mask, torch.cat([mask, mask]))


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


def test_concatenate_rejects_mismatched_patch_dtype() -> None:
    first = _make_features(batch_size=1)
    second = _make_features(batch_size=1)
    second = replace(second, patch_tokens=second.patch_tokens.double())
    with pytest.raises(ValueError, match="patch-token dtype"):
        concatenate_encoder_features([first, second])


def test_concatenate_rejects_mismatched_cls_presence() -> None:
    first = _make_features(batch_size=1, cls_dim=5)
    second = _make_features(batch_size=1, cls_dim=None)
    with pytest.raises(ValueError, match="cls_tokens is present"):
        concatenate_encoder_features([first, second])


def test_concatenate_rejects_mismatched_cls_dimension() -> None:
    first = _make_features(batch_size=1, cls_dim=5)
    second = _make_features(batch_size=1, cls_dim=6)
    with pytest.raises(ValueError, match="CLS dimension"):
        concatenate_encoder_features([first, second])


def test_concatenate_rejects_mismatched_cls_dtype() -> None:
    first = _make_features(batch_size=1, cls_dim=5)
    second = _make_features(batch_size=1, cls_dim=5)
    assert second.cls_tokens is not None
    second = replace(second, cls_tokens=second.cls_tokens.double())
    with pytest.raises(ValueError, match="CLS-token dtype"):
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


def test_select_one_rejects_out_of_range_index() -> None:
    batch = _make_features(batch_size=2)
    with pytest.raises(ValueError, match=r"index must be in \[0, 2\)"):
        select_one(batch, 2)


def test_select_one_returns_a_length_one_batch() -> None:
    batch = _make_features(batch_size=3, num_patches=6, patch_grid=[(2, 3), (2, 3), (2, 3)])
    one = select_one(batch, 1)
    assert one.batch_size == 1
    assert torch.equal(one.patch_tokens[0], batch.patch_tokens[1])
    assert one.image_sizes == [batch.image_sizes[1]]


def test_select_one_trims_padding_to_the_samples_own_patch_count() -> None:
    small = _make_features(batch_size=1, num_patches=4, patch_grid=[(2, 2)])
    large = _make_features(batch_size=1, num_patches=6, patch_grid=[(2, 3)])
    padded_batch = concatenate_encoder_features([small, large])  # small is padded to 6 patches

    selected = select_one(padded_batch, 0)

    assert selected.patch_tokens.shape == (1, 4, small.patch_tokens.shape[-1])
    assert selected.valid_patch_mask is None
    assert torch.equal(selected.patch_tokens[0], small.patch_tokens[0])


def test_select_one_preserves_cls_tokens_when_present() -> None:
    batch = _make_features(batch_size=2, cls_dim=5)
    assert batch.cls_tokens is not None
    selected = select_one(batch, 1)
    assert selected.cls_tokens is not None
    assert torch.equal(selected.cls_tokens[0], batch.cls_tokens[1])


def test_select_one_uses_the_valid_mask_instead_of_assuming_prefix_padding() -> None:
    tokens = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
    mask = torch.tensor([[False, True, False, True]])
    batch = EncoderFeatures(
        patch_tokens=tokens,
        cls_tokens=None,
        patch_grid=torch.tensor([[1, 2]]),
        valid_patch_mask=mask,
        image_sizes=[(224, 224)],
        encoder_model="fake",
        encoder_revision=None,
    )

    selected = select_one(batch, 0)

    assert torch.equal(selected.patch_tokens[0], tokens[0, [1, 3]])


def test_select_one_handles_no_cls_tokens() -> None:
    batch = _make_features(batch_size=2, cls_dim=None)
    selected = select_one(batch, 0)
    assert selected.cls_tokens is None


def test_to_moves_patch_tokens_patch_grid_and_cls_tokens() -> None:
    features = _make_features(cls_dim=5)
    moved = features.to("cpu")
    assert moved.patch_tokens.device == torch.device("cpu")
    assert moved.patch_grid.device == torch.device("cpu")
    assert moved.cls_tokens is not None
    assert moved.cls_tokens.device == torch.device("cpu")


def test_to_leaves_cls_tokens_none_when_absent() -> None:
    features = _make_features(cls_dim=None)
    moved = features.to("cpu")
    assert moved.cls_tokens is None


def test_to_moves_valid_patch_mask_when_present() -> None:
    mask = torch.tensor([[True, True, True, True, True, True], [True, True, True, True, False, False]])
    features = _make_features(num_patches=6, patch_grid=[(2, 3), (2, 2)], valid_patch_mask=mask)
    moved = features.to("cpu")
    assert moved.valid_patch_mask is not None
    assert moved.valid_patch_mask.device == torch.device("cpu")


def test_to_leaves_valid_patch_mask_none_when_absent() -> None:
    features = _make_features(valid_patch_mask=None)
    moved = features.to("cpu")
    assert moved.valid_patch_mask is None


def test_to_preserves_values_and_non_tensor_fields() -> None:
    features = _make_features()
    moved = features.to("cpu")
    assert torch.equal(moved.patch_tokens, features.patch_tokens)
    assert moved.image_sizes == features.image_sizes
    assert moved.encoder_model == features.encoder_model
    assert moved.encoder_revision == features.encoder_revision
    assert moved.metadata == features.metadata


def test_to_accepts_a_torch_device_object() -> None:
    features = _make_features()
    moved = features.to(torch.device("cpu"))
    assert moved.patch_tokens.device == torch.device("cpu")
