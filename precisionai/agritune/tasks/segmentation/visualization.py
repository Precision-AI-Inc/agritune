# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Turning per-pixel class predictions into human-viewable images.

Pure image composition — no I/O beyond what callers pass in and get back, so it stays usable from
both :mod:`~precisionai.agritune.services.prediction_service` and ad hoc notebook/CLI use.
"""

import colorsys

import numpy as np
import torch
from PIL import Image


def default_palette(num_classes: int) -> np.ndarray:
    """Return a deterministic, visually distinct RGB color per class.

    Colors are spaced evenly around the HSV hue wheel, so adjacent class indices stay visually
    distinguishable regardless of ``num_classes``.

    Parameters
    ----------
    num_classes : int

    Returns
    -------
    numpy.ndarray
        Shape ``(num_classes, 3)``, ``uint8``.
    """
    colors = []
    for class_index in range(num_classes):
        hue = class_index / max(num_classes, 1)
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
        colors.append((round(red * 255), round(green * 255), round(blue * 255)))
    return np.array(colors, dtype=np.uint8)


def colorize_predictions(
    predictions: torch.Tensor, *, num_classes: int, palette: np.ndarray | None = None
) -> np.ndarray:
    """Map a per-pixel class-index map to an RGB image.

    Parameters
    ----------
    predictions : torch.Tensor
        Shape ``(H, W)``, integer class indices in ``[0, num_classes)``.
    num_classes : int
    palette : numpy.ndarray | None, optional
        Shape ``(num_classes, 3)``, ``uint8``; defaults to :func:`default_palette`.

    Returns
    -------
    numpy.ndarray
        Shape ``(H, W, 3)``, ``uint8``.
    """
    resolved_palette = default_palette(num_classes) if palette is None else palette
    indices = predictions.detach().cpu().numpy().astype(np.int64)
    return resolved_palette[indices]


def overlay_predictions_on_image(
    image: Image.Image,
    predictions: torch.Tensor,
    *,
    num_classes: int,
    alpha: float = 0.5,
    palette: np.ndarray | None = None,
) -> Image.Image:
    """Alpha-blend a colorized prediction map over the original image.

    The prediction map is nearest-neighbor resized to ``image``'s resolution before blending, so
    predictions computed at a different (typically lower, patch-grid-derived) resolution still
    align with the original image without smearing class boundaries.

    Parameters
    ----------
    image : PIL.Image.Image
        The original sample image.
    predictions : torch.Tensor
        Shape ``(H, W)``, integer class indices.
    num_classes : int
    alpha : float, optional
        Overlay opacity in ``[0, 1]``; ``0`` is the original image, ``1`` is the color map alone.
    palette : numpy.ndarray | None, optional
        Shape ``(num_classes, 3)``, ``uint8``; defaults to :func:`default_palette`.

    Returns
    -------
    PIL.Image.Image

    Raises
    ------
    ValueError
        If ``alpha`` is outside ``[0, 1]``.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1]; got {alpha}")

    color_map = colorize_predictions(predictions, num_classes=num_classes, palette=palette)
    overlay = Image.fromarray(color_map, mode="RGB").resize(image.size, resample=Image.Resampling.NEAREST)
    return Image.blend(image.convert("RGB"), overlay, alpha)


def side_by_side(*panels: Image.Image) -> Image.Image:
    """Compose images left-to-right into one panel, for quick visual comparison.

    Every panel is resized to the first panel's height (aspect-preserving) before compositing.

    Parameters
    ----------
    *panels : PIL.Image.Image
        At least one image, e.g. ``(original, ground_truth_overlay, prediction_overlay)``.

    Returns
    -------
    PIL.Image.Image

    Raises
    ------
    ValueError
        If no panels are given.
    """
    if not panels:
        raise ValueError("side_by_side requires at least one panel")

    target_height = panels[0].height
    resized = []
    for panel in panels:
        width = round(panel.width * target_height / panel.height)
        resized.append(panel.resize((width, target_height)))

    total_width = sum(panel.width for panel in resized)
    canvas = Image.new("RGB", (total_width, target_height))
    x_offset = 0
    for panel in resized:
        canvas.paste(panel, (x_offset, 0))
        x_offset += panel.width
    return canvas
