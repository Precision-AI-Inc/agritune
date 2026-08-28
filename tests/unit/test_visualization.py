# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.tasks.segmentation.visualization."""

import numpy as np
import pytest
import torch
from PIL import Image

from precisionai.agritune.tasks.segmentation.visualization import (
    colorize_predictions,
    default_palette,
    overlay_predictions_on_image,
    side_by_side,
)


def test_default_palette_has_one_distinct_color_per_class() -> None:
    palette = default_palette(4)
    assert palette.shape == (4, 3)
    assert palette.dtype == np.uint8
    assert len({tuple(color) for color in palette.tolist()}) == 4


def test_colorize_predictions_maps_each_class_to_its_palette_color() -> None:
    predictions = torch.tensor([[0, 1], [1, 0]])
    palette = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)
    colorized = colorize_predictions(predictions, num_classes=2, palette=palette)
    assert colorized.shape == (2, 2, 3)
    assert colorized[0, 0].tolist() == [10, 20, 30]
    assert colorized[0, 1].tolist() == [40, 50, 60]


def test_colorize_predictions_uses_default_palette_when_none_given() -> None:
    predictions = torch.zeros(2, 2, dtype=torch.long)
    colorized = colorize_predictions(predictions, num_classes=3)
    assert colorized.shape == (2, 2, 3)


def test_overlay_predictions_resizes_to_image_resolution() -> None:
    image = Image.new("RGB", (8, 8), color=(0, 0, 0))
    predictions = torch.zeros(2, 2, dtype=torch.long)  # coarser than the 8x8 image
    overlay = overlay_predictions_on_image(image, predictions, num_classes=2)
    assert overlay.size == (8, 8)
    assert overlay.mode == "RGB"


def test_overlay_alpha_zero_returns_the_original_image() -> None:
    image = Image.new("RGB", (4, 4), color=(100, 150, 200))
    predictions = torch.ones(4, 4, dtype=torch.long)
    overlay = overlay_predictions_on_image(image, predictions, num_classes=2, alpha=0.0)
    assert np.array(overlay).tolist() == np.array(image).tolist()


def test_overlay_alpha_one_returns_the_pure_color_map() -> None:
    palette = np.array([[0, 0, 0], [200, 100, 50]], dtype=np.uint8)
    image = Image.new("RGB", (4, 4), color=(0, 0, 0))
    predictions = torch.ones(4, 4, dtype=torch.long)
    overlay = overlay_predictions_on_image(image, predictions, num_classes=2, alpha=1.0, palette=palette)
    assert np.array(overlay)[0, 0].tolist() == [200, 100, 50]


def test_overlay_invalid_alpha_raises() -> None:
    image = Image.new("RGB", (2, 2))
    predictions = torch.zeros(2, 2, dtype=torch.long)
    with pytest.raises(ValueError, match="alpha must be in"):
        overlay_predictions_on_image(image, predictions, num_classes=1, alpha=1.5)


def test_side_by_side_concatenates_panels_horizontally() -> None:
    left = Image.new("RGB", (4, 8), color=(255, 0, 0))
    right = Image.new("RGB", (4, 8), color=(0, 255, 0))
    combined = side_by_side(left, right)
    assert combined.size == (8, 8)
    assert combined.getpixel((0, 0)) == (255, 0, 0)
    assert combined.getpixel((7, 0)) == (0, 255, 0)


def test_side_by_side_resizes_panels_to_first_panels_height() -> None:
    tall = Image.new("RGB", (4, 10))
    short = Image.new("RGB", (4, 5))
    combined = side_by_side(tall, short)
    assert combined.height == 10


def test_side_by_side_requires_at_least_one_panel() -> None:
    with pytest.raises(ValueError, match="at least one panel"):
        side_by_side()
