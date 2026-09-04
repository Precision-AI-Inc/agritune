# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Dataset orchestration — powers ``agritune dataset validate``/``inspect``/``init`` and the API layer."""

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from precisionai.agritune.data.manifest import parse_manifest_rows
from precisionai.agritune.data.validation import ValidationReport, validate_manifest
from precisionai.agritune.logging import get_logger
from precisionai.agritune.utils.scaffold import read_scaffold_template, write_scaffold_file

logger = get_logger(__name__)


def validate_dataset(
    manifest_path: str, *, num_classes: int | None = None, ignore_index: int | None = None
) -> ValidationReport:
    """Validate a dataset manifest — see :func:`~precisionai.agritune.data.validation.validate_manifest`."""
    report = validate_manifest(manifest_path, num_classes=num_classes, ignore_index=ignore_index)
    if report.is_valid:
        logger.info("manifest %s is valid", manifest_path)
    else:
        logger.warning("manifest %s has %d issue(s)", manifest_path, len(report.issues))
    return report


def inspect_dataset(manifest_path: str) -> dict[str, Any]:
    """Report basic statistics about a dataset manifest.

    Parameters
    ----------
    manifest_path : str
        Path to the manifest CSV.

    Returns
    -------
    dict[str, Any]
        ``num_samples``, ``metadata_columns`` (every extra manifest column seen), and
        ``class_pixel_counts`` (a label-value -> pixel-count histogram across every readable
        mask, skipping any that fail to open).
    """
    rows = parse_manifest_rows(manifest_path)
    base_dir_columns: set[str] = set()
    for row in rows:
        base_dir_columns.update(row.metadata.keys())

    class_pixel_counts: Counter[int] = Counter()
    base_dir = Path(manifest_path).parent
    for row in rows:
        mask_path = base_dir / row.mask_path
        try:
            with Image.open(mask_path) as mask:
                labels, counts = np.unique(np.array(mask), return_counts=True)
        except OSError:
            continue
        class_pixel_counts.update(dict(zip((int(label) for label in labels), (int(c) for c in counts), strict=True)))

    logger.info("inspected manifest %s: %d sample(s)", manifest_path, len(rows))
    return {
        "num_samples": len(rows),
        "metadata_columns": sorted(base_dir_columns),
        "class_pixel_counts": dict(sorted(class_pixel_counts.items())),
    }


def render_manifest_template() -> str:
    """Return the packaged example dataset manifest CSV's text.

    Returns
    -------
    str
    """
    return read_scaffold_template("manifest.csv")


def write_manifest_template(output_path: str | Path, *, force: bool = False) -> Path:
    """Write the example dataset manifest CSV to ``output_path``.

    Parameters
    ----------
    output_path : str | Path
        Destination file path. Parent directories are created as needed.
    force : bool, optional
        Overwrite ``output_path`` if it already exists (default ``False``).

    Returns
    -------
    Path
        ``output_path``.

    Raises
    ------
    FileExistsError
        If ``output_path`` already exists and ``force`` is ``False``.
    """
    path = write_scaffold_file(output_path, render_manifest_template(), force=force)
    logger.info("wrote manifest template to %s", path)
    return path
