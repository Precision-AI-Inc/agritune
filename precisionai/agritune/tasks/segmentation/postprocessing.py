# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Turning decoder logits into per-pixel class predictions, at any target resolution."""

import torch
from torch.nn import functional


def logits_to_predictions(logits: torch.Tensor) -> torch.Tensor:
    """Convert per-class logits to a per-pixel class-index map.

    Parameters
    ----------
    logits : torch.Tensor
        Shape ``(B, num_classes, H, W)``.

    Returns
    -------
    torch.Tensor
        Shape ``(B, H, W)``, integer class indices.
    """
    return logits.argmax(dim=1)


def resize_predictions(predictions: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Nearest-neighbor resize a per-pixel class-index map — never interpolate label values.

    Parameters
    ----------
    predictions : torch.Tensor
        Shape ``(B, H, W)``, integer class indices.
    size : tuple[int, int]
        Target ``(height, width)``.

    Returns
    -------
    torch.Tensor
        Shape ``(B, *size)``, integer class indices.
    """
    resized = functional.interpolate(predictions.unsqueeze(1).to(torch.float32), size=size, mode="nearest")
    return resized.squeeze(1).to(predictions.dtype)
