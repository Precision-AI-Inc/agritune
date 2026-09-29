# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.tasks.segmentation.upsample."""

import pytest
import torch
from torch.nn import functional

from precisionai.agritune.tasks.segmentation.upsample import bilinear_resize, interpolation_matrix


def _reference(inputs: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return functional.interpolate(inputs, size=size, mode="bilinear", align_corners=False)


@pytest.mark.parametrize(
    ("in_hw", "out_hw"),
    [
        ((16, 16), (224, 224)),  # square upsample, the linear-probe case
        ((38, 57), (224, 224)),  # non-square grid to square mask
        ((5, 7), (13, 3)),  # mixed: upsample rows, downsample columns
        ((9, 4), (3, 2)),  # pure downsample
        ((1, 1), (6, 4)),  # global-pooled map broadcast back (ASPP/PPM image pooling)
    ],
)
def test_matches_functional_interpolate_forward(in_hw: tuple[int, int], out_hw: tuple[int, int]) -> None:
    generator = torch.Generator().manual_seed(0)
    inputs = torch.randn(2, 3, *in_hw, generator=generator, dtype=torch.float64)

    result = bilinear_resize(inputs, out_hw)

    assert result.shape == (2, 3, *out_hw)
    torch.testing.assert_close(result, _reference(inputs, out_hw), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(("in_hw", "out_hw"), [((16, 16), (64, 48)), ((7, 5), (3, 11))])
def test_gradients_match_functional_interpolate(in_hw: tuple[int, int], out_hw: tuple[int, int]) -> None:
    generator = torch.Generator().manual_seed(1)
    inputs = torch.randn(2, 4, *in_hw, generator=generator, dtype=torch.float64)
    upstream = torch.randn(2, 4, *out_hw, generator=generator, dtype=torch.float64)

    ours = inputs.clone().requires_grad_(True)
    (bilinear_resize(ours, out_hw) * upstream).sum().backward()
    theirs = inputs.clone().requires_grad_(True)
    (_reference(theirs, out_hw) * upstream).sum().backward()

    assert ours.grad is not None
    assert theirs.grad is not None
    torch.testing.assert_close(ours.grad, theirs.grad, rtol=1e-12, atol=1e-12)


def test_float32_matches_functional_interpolate_within_float32_rounding() -> None:
    inputs = torch.randn(4, 15, 16, 16, generator=torch.Generator().manual_seed(2))

    torch.testing.assert_close(bilinear_resize(inputs, (224, 224)), _reference(inputs, (224, 224)))


def test_same_size_returns_the_input_unchanged() -> None:
    inputs = torch.randn(1, 2, 5, 6)

    assert bilinear_resize(inputs, (5, 6)) is inputs


def test_accepts_a_torch_size() -> None:
    inputs = torch.randn(1, 1, 3, 3)
    target = torch.zeros(1, 7, 9)

    assert bilinear_resize(inputs, target.shape[-2:]).shape == (1, 1, 7, 9)


def test_preserves_dtype_and_device() -> None:
    inputs = torch.randn(1, 2, 4, 4, dtype=torch.float64)

    result = bilinear_resize(inputs, (8, 8))

    assert result.dtype == torch.float64
    assert result.device == inputs.device


def test_rows_of_an_upsampling_matrix_sum_to_one() -> None:
    matrix = interpolation_matrix(16, 224, device=torch.device("cpu"), dtype=torch.float64)

    assert matrix.shape == (224, 16)
    torch.testing.assert_close(matrix.sum(dim=1), torch.ones(224, dtype=torch.float64))


def test_interpolation_matrix_is_cached_per_size_device_and_dtype() -> None:
    first = interpolation_matrix(8, 20, device=torch.device("cpu"), dtype=torch.float32)
    second = interpolation_matrix(8, 20, device=torch.device("cpu"), dtype=torch.float32)
    other_dtype = interpolation_matrix(8, 20, device=torch.device("cpu"), dtype=torch.float64)

    assert first is second
    assert other_dtype is not first


@pytest.mark.parametrize(("in_size", "out_size"), [(0, 4), (4, 0), (-1, 2)])
def test_interpolation_matrix_rejects_non_positive_sizes(in_size: int, out_size: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        interpolation_matrix(in_size, out_size, device=torch.device("cpu"), dtype=torch.float32)


def test_rejects_non_4d_input() -> None:
    with pytest.raises(ValueError, match=r"shape \(B, C, H, W\)"):
        bilinear_resize(torch.randn(2, 4, 4), (8, 8))


def test_rejects_integer_input() -> None:
    with pytest.raises(ValueError, match="floating point"):
        bilinear_resize(torch.zeros(1, 1, 2, 2, dtype=torch.long), (4, 4))


def test_rejects_a_size_without_exactly_two_entries() -> None:
    with pytest.raises(ValueError, match=r"size must be \(height, width\)"):
        bilinear_resize(torch.randn(1, 1, 2, 2), (4, 4, 4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA-enabled machine")
def test_matches_functional_interpolate_on_cuda() -> None:
    device = torch.device("cuda")
    inputs = torch.randn(8, 15, 16, 16, device=device, dtype=torch.float64)

    torch.testing.assert_close(bilinear_resize(inputs, (224, 224)), _reference(inputs, (224, 224)))
