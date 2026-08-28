# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""CSV manifest parsing for agricultural segmentation datasets.

A manifest is a CSV file with at least ``sample_id``, ``image_path``, ``mask_path`` columns.
Any additional columns become per-sample metadata (e.g. ``field_id``, ``farm_id``,
``capture_date``) — see ``docs/datasets.md``.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REQUIRED_COLUMNS = ("sample_id", "image_path", "mask_path")


@dataclass
class ManifestRow:
    """One row of a dataset manifest.

    Attributes
    ----------
    sample_id : str
        Unique identifier for this sample.
    image_path : str
        Path to the image file, relative to the manifest's own directory.
    mask_path : str
        Path to the segmentation mask file, relative to the manifest's own directory.
    metadata : dict[str, Any]
        Every other manifest column, keyed by column name. Empty string values are omitted.
    """

    sample_id: str
    image_path: str
    mask_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_manifest_rows(path: str | Path) -> list[ManifestRow]:
    """Parse a manifest CSV into rows, without checking for duplicate sample IDs.

    Parameters
    ----------
    path : str | Path
        Path to the manifest CSV file.

    Returns
    -------
    list[ManifestRow]
        One entry per data row, in file order. May contain duplicate ``sample_id`` values — use
        :func:`load_manifest` when a unique index is required, or
        ``precisionai.agritune.data.validation.validate_manifest`` to report duplicates.

    Raises
    ------
    ValueError
        If the file has no header row, or is missing a required column.
    """
    manifest_path = Path(path)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"manifest {manifest_path} has no header row")
        missing = [column for column in _REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"manifest {manifest_path} is missing required column(s): {', '.join(missing)}")
        extra_columns = [column for column in reader.fieldnames if column not in _REQUIRED_COLUMNS]

        rows = []
        for raw_row in reader:
            metadata = {column: raw_row[column] for column in extra_columns if raw_row.get(column)}
            rows.append(
                ManifestRow(
                    sample_id=raw_row["sample_id"],
                    image_path=raw_row["image_path"],
                    mask_path=raw_row["mask_path"],
                    metadata=metadata,
                )
            )
        return rows


def load_manifest(path: str | Path) -> list[ManifestRow]:
    """Parse a manifest CSV and reject duplicate sample IDs.

    Parameters
    ----------
    path : str | Path
        Path to the manifest CSV file.

    Returns
    -------
    list[ManifestRow]
        One entry per data row, in file order, with unique ``sample_id`` values guaranteed.

    Raises
    ------
    ValueError
        If the file is malformed (see :func:`parse_manifest_rows`), or contains a duplicate
        ``sample_id``.
    """
    rows = parse_manifest_rows(path)
    seen: dict[str, int] = {}
    for row in rows:
        seen[row.sample_id] = seen.get(row.sample_id, 0) + 1
    duplicates = sorted(sample_id for sample_id, count in seen.items() if count > 1)
    if duplicates:
        raise ValueError(f"manifest {path} contains duplicate sample_id(s): {', '.join(duplicates)}")
    return rows
