# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Bilinear resizing expressed as two small matrix products instead of ``F.interpolate``.

``functional.interpolate(mode="bilinear", align_corners=False)`` is a separable linear map: every
output row is a fixed weighted sum of input rows, and likewise for columns. Writing it as
``A_h @ x @ A_w.T`` computes exactly the same values, but its backward pass is also two matrix
products. The native CUDA backward of ``upsample_bilinear2d`` instead scatters gradients with
atomic adds, which serializes heavily when a small patch grid (e.g. 16x16) is upsampled to a large
mask (e.g. 224x224): every source pixel receives hundreds of colliding writes. On an A100 that
backward dominated a linear-probe training step (about 48 of 61 ms per 256-sample batch); the
matrix form brings the whole step to about 8 ms.

The interpolation matrices are derived by running ``functional.interpolate`` itself on an identity
basis, so they reproduce PyTorch's exact sampling convention (half-pixel centers, edge clamping,
downsampling included) rather than re-implementing it.
"""

from collections.abc import Sequence
from functools import lru_cache

import torch
from torch.nn import functional


@lru_cache(maxsize=64)
def _interpolation_matrix_cpu(in_size: int, out_size: int) -> torch.Tensor:
    identity = torch.eye(in_size, dtype=torch.float64).reshape(in_size, 1, in_size, 1)
    columns = functional.interpolate(identity, size=(out_size, 1), mode="bilinear", align_corners=False)
    return columns[:, 0, :, 0].T.contiguous()


@lru_cache(maxsize=256)
def _interpolation_matrix(in_size: int, out_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return _interpolation_matrix_cpu(in_size, out_size).to(device=device, dtype=dtype)


def interpolation_matrix(in_size: int, out_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return the ``(out_size, in_size)`` matrix that bilinearly resizes one axis.

    Parameters
    ----------
    in_size : int
        Length of the axis being resized.
    out_size : int
        Target length of that axis.
    device : torch.device
    dtype : torch.dtype
        Floating-point dtype of the returned matrix.

    Returns
    -------
    torch.Tensor of shape (out_size, in_size)
        Row ``i`` holds the weights that combine input positions into output position ``i``,
        matching ``functional.interpolate(mode="bilinear", align_corners=False)``.

    Raises
    ------
    ValueError
        If either size is not positive.
    """
    if in_size < 1 or out_size < 1:
        raise ValueError(f"in_size and out_size must be positive; got {in_size} and {out_size}")
    return _interpolation_matrix(in_size, out_size, device, dtype)


def bilinear_resize(inputs: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    """Resize a ``(B, C, H, W)`` tensor to ``size`` with bilinear interpolation.

    Numerically equivalent to ``functional.interpolate(inputs, size=size, mode="bilinear",
    align_corners=False)`` up to floating-point rounding, and differentiable, but with a backward
    pass made of matrix products rather than atomic scatter-adds (see the module docstring).

    Parameters
    ----------
    inputs : torch.Tensor of shape (B, C, H, W)
        Floating-point input, e.g. per-patch logits on the patch grid.
    size : Sequence[int]
        Target ``(height, width)`` — a tuple, list, or ``torch.Size`` of length 2.

    Returns
    -------
    torch.Tensor of shape (B, C, size[0], size[1])
        Same dtype and device as ``inputs``. Returned unchanged (the same tensor) when ``size``
        already equals ``(H, W)``, which bilinear resizing with ``align_corners=False`` maps to an
        exact copy anyway.

    Raises
    ------
    ValueError
        If ``inputs`` is not 4-dimensional or not floating point, or ``size`` does not have
        exactly two entries.
    """
    if len(size) != 2:
        raise ValueError(f"size must be (height, width); got {tuple(size)}")
    if inputs.ndim != 4:
        raise ValueError(f"inputs must have shape (B, C, H, W); got {tuple(inputs.shape)}")
    if not inputs.is_floating_point():
        raise ValueError(f"inputs must be floating point; got {inputs.dtype}")
    out_height, out_width = int(size[0]), int(size[1])
    in_height, in_width = inputs.shape[-2:]
    if (in_height, in_width) == (out_height, out_width):
        return inputs
    rows = interpolation_matrix(in_height, out_height, device=inputs.device, dtype=inputs.dtype)
    columns = interpolation_matrix(in_width, out_width, device=inputs.device, dtype=inputs.dtype)
    return torch.einsum("oh,bchw,pw->bcop", rows, inputs, columns)
