# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Builds a tiny on-disk manifest + image/mask pairs for dataset/split/validation tests."""

from pathlib import Path

import numpy as np
from PIL import Image

# sample_id, field_id, mask fill value
_DEFAULT_ROWS = [
    ("sample-0", "field-a", 0),
    ("sample-1", "field-a", 1),
    ("sample-2", "field-b", 2),
    ("sample-3", "field-b", 0),
]


def build_manifest(
    root: Path,
    *,
    rows: list[tuple[str, str, int]] | None = None,
    image_size: tuple[int, int] = (4, 4),
    mismatched_mask_size_for: str | None = None,
    omit_image_for: str | None = None,
    omit_mask_for: str | None = None,
) -> Path:
    """Write a manifest CSV plus tiny RGB image / single-channel mask PNGs under ``root``.

    Parameters
    ----------
    root : Path
        Directory to write the manifest and image/mask files into (typically ``tmp_path``).
    rows : list[tuple[str, str, int]] | None, optional
        ``(sample_id, field_id, mask_fill_value)`` tuples; defaults to :data:`_DEFAULT_ROWS`.
    image_size : tuple[int, int], optional
        ``(width, height)`` for generated images/masks.
    mismatched_mask_size_for : str | None, optional
        If set, write a mask one pixel larger for this sample_id, to simulate a dimension
        mismatch.
    omit_image_for : str | None, optional
        If set, skip writing the image file for this sample_id, to simulate a missing image.
    omit_mask_for : str | None, optional
        If set, skip writing the mask file for this sample_id, to simulate a missing mask.

    Returns
    -------
    Path
        Path to the written ``manifest.csv``.
    """
    rows = rows if rows is not None else _DEFAULT_ROWS
    lines = ["sample_id,image_path,mask_path,field_id"]

    for sample_id, field_id, fill_value in rows:
        image_path = f"{sample_id}_image.png"
        mask_path = f"{sample_id}_mask.png"
        lines.append(f"{sample_id},{image_path},{mask_path},{field_id}")

        if sample_id != omit_image_for:
            image = Image.fromarray(np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8))
            image.save(root / image_path)

        if sample_id != omit_mask_for:
            mask_size = image_size
            if sample_id == mismatched_mask_size_for:
                mask_size = (image_size[0] + 1, image_size[1] + 1)
            mask = Image.fromarray(np.full((mask_size[1], mask_size[0]), fill_value, dtype=np.uint8))
            mask.save(root / mask_path)

    manifest_path = root / "manifest.csv"
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path
