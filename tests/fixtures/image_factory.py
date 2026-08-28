# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Builds tiny deterministic PIL image/mask pairs for augmentation tests."""

import numpy as np
from PIL import Image


def make_image(size: tuple[int, int] = (8, 6)) -> Image.Image:
    """Return an ``(width, height) = size`` RGB gradient image (not flat, so transforms are visible)."""
    width, height = size
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    grid = np.stack(np.meshgrid(x, y), axis=-1)  # (height, width, 2)
    array = np.dstack([grid[..., 0], grid[..., 1], np.full((height, width), 128, dtype=np.uint8)])
    return Image.fromarray(array, mode="RGB")


def make_mask(size: tuple[int, int] = (8, 6), *, fill_value: int = 1) -> Image.Image:
    """Return an ``(width, height) = size`` single-channel mask filled with ``fill_value``."""
    width, height = size
    array = np.full((height, width), fill_value, dtype=np.uint8)
    return Image.fromarray(array, mode="L")
