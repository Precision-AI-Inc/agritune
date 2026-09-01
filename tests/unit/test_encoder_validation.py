# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.encoder.validation."""

import pytest
import torch

from precisionai.agritune.encoder.validation import EncoderConsistencyError, EncoderResponseValidator
from precisionai.agritune.schemas.features import EncoderFeatures


def _features(
    *,
    patch_dim: int,
    cls_dim: int | None,
    patch_dtype: torch.dtype = torch.float32,
    cls_dtype: torch.dtype = torch.float32,
) -> EncoderFeatures:
    return EncoderFeatures(
        patch_tokens=torch.randn(1, 4, patch_dim, dtype=patch_dtype),
        cls_tokens=torch.randn(1, cls_dim, dtype=cls_dtype) if cls_dim is not None else None,
        patch_grid=torch.tensor([[2, 2]]),
        valid_patch_mask=None,
        image_sizes=[(224, 224)],
        encoder_model="pai-embedding",
        encoder_revision=None,
    )


def test_first_call_establishes_fingerprint_without_raising() -> None:
    validator = EncoderResponseValidator()
    validator.validate(_features(patch_dim=8, cls_dim=5))  # should not raise


def test_matching_dimensions_across_calls_do_not_raise() -> None:
    validator = EncoderResponseValidator()
    validator.validate(_features(patch_dim=8, cls_dim=5))
    validator.validate(_features(patch_dim=8, cls_dim=5))  # should not raise


def test_patch_dim_drift_raises() -> None:
    validator = EncoderResponseValidator()
    validator.validate(_features(patch_dim=8, cls_dim=5))
    with pytest.raises(EncoderConsistencyError, match="dimensions changed mid-run"):
        validator.validate(_features(patch_dim=16, cls_dim=5))


def test_cls_dim_drift_raises() -> None:
    validator = EncoderResponseValidator()
    validator.validate(_features(patch_dim=8, cls_dim=5))
    with pytest.raises(EncoderConsistencyError, match="dimensions changed mid-run"):
        validator.validate(_features(patch_dim=8, cls_dim=6))


def test_cls_presence_drift_raises() -> None:
    validator = EncoderResponseValidator()
    validator.validate(_features(patch_dim=8, cls_dim=5))
    with pytest.raises(EncoderConsistencyError, match="dimensions changed mid-run"):
        validator.validate(_features(patch_dim=8, cls_dim=None))


def test_patch_dtype_drift_raises() -> None:
    validator = EncoderResponseValidator()
    validator.validate(_features(patch_dim=8, cls_dim=5))
    with pytest.raises(EncoderConsistencyError, match="dimensions changed mid-run"):
        validator.validate(_features(patch_dim=8, cls_dim=5, patch_dtype=torch.float64))


def test_cls_dtype_drift_raises() -> None:
    validator = EncoderResponseValidator()
    validator.validate(_features(patch_dim=8, cls_dim=5))
    with pytest.raises(EncoderConsistencyError, match="dimensions changed mid-run"):
        validator.validate(_features(patch_dim=8, cls_dim=5, cls_dtype=torch.float64))
