# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Structural validation for a dataset manifest — powers ``agritune dataset validate``.

Checks: missing images, missing masks, duplicate IDs, image/mask dimension mismatch, and
(optionally, when ``num_classes`` is known) invalid label values in the mask.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from precisionai.agritune.data.manifest import ManifestRow, parse_manifest_rows


@dataclass
class ValidationIssue:
    """One problem found while validating a manifest.

    Attributes
    ----------
    sample_id : str | None
        The affected sample, or ``None`` for a manifest-wide issue.
    category : str
        Machine-readable issue category, e.g. ``"missing_image"``, ``"duplicate_id"``,
        ``"dimension_mismatch"``, ``"invalid_label"``.
    message : str
        Human-readable description.
    """

    sample_id: str | None
    category: str
    message: str


@dataclass
class ValidationReport:
    """The result of validating a manifest.

    Attributes
    ----------
    issues : list[ValidationIssue]
        All problems found, in the order they were detected.
    """

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Return ``True`` if no issues were found."""
        return not self.issues

    def issues_by_category(self) -> dict[str, list[ValidationIssue]]:
        """Group :attr:`issues` by their ``category``."""
        grouped: dict[str, list[ValidationIssue]] = {}
        for issue in self.issues:
            grouped.setdefault(issue.category, []).append(issue)
        return grouped


def validate_manifest(
    path: str | Path,
    *,
    check_image_mask_dimensions: bool = True,
    num_classes: int | None = None,
    ignore_index: int | None = None,
) -> ValidationReport:
    """Validate a dataset manifest and its referenced files.

    Parameters
    ----------
    path : str | Path
        Path to the manifest CSV. Image/mask paths are resolved relative to its directory.
    check_image_mask_dimensions : bool, optional
        When ``True``, open each existing image/mask pair and flag a size mismatch.
    num_classes : int | None, optional
        When given, flag any mask pixel value outside ``[0, num_classes)`` (excluding
        ``ignore_index``) as an ``"invalid_label"`` issue.
    ignore_index : int | None, optional
        A label value to exclude from the ``num_classes`` range check (e.g. a "void"/ignore
        class).

    Returns
    -------
    ValidationReport
        All issues found. An empty report (``report.is_valid``) means the manifest is well-formed
        and every referenced file exists and is readable.
    """
    manifest_path = Path(path)
    base_dir = manifest_path.parent
    rows = parse_manifest_rows(manifest_path)
    report = ValidationReport()

    report.issues.extend(_find_duplicate_ids(rows))
    for row in rows:
        report.issues.extend(
            _validate_row_files(
                row,
                base_dir,
                check_image_mask_dimensions=check_image_mask_dimensions,
                num_classes=num_classes,
                ignore_index=ignore_index,
            )
        )

    return report


def _find_duplicate_ids(rows: list[ManifestRow]) -> list[ValidationIssue]:
    seen_counts: dict[str, int] = {}
    for row in rows:
        seen_counts[row.sample_id] = seen_counts.get(row.sample_id, 0) + 1
    return [
        ValidationIssue(sample_id, "duplicate_id", f"sample_id '{sample_id}' appears {count} times")
        for sample_id, count in seen_counts.items()
        if count > 1
    ]


def _validate_row_files(
    row: ManifestRow,
    base_dir: Path,
    *,
    check_image_mask_dimensions: bool,
    num_classes: int | None,
    ignore_index: int | None,
) -> list[ValidationIssue]:
    image_path = base_dir / row.image_path
    mask_path = base_dir / row.mask_path
    image_exists = image_path.is_file()
    mask_exists = mask_path.is_file()

    issues: list[ValidationIssue] = []
    if not image_exists:
        issues.append(ValidationIssue(row.sample_id, "missing_image", f"image not found: {image_path}"))
    if not mask_exists:
        issues.append(ValidationIssue(row.sample_id, "missing_mask", f"mask not found: {mask_path}"))
    if not (image_exists and mask_exists):
        return issues

    try:
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if check_image_mask_dimensions and image.size != mask.size:
                issues.append(
                    ValidationIssue(
                        row.sample_id, "dimension_mismatch", f"image size {image.size} != mask size {mask.size}"
                    )
                )
            if num_classes is not None:
                mask_array = np.array(mask)
                invalid = [
                    int(label)
                    for label in np.unique(mask_array)
                    if not (0 <= label < num_classes) and label != ignore_index
                ]
                if invalid:
                    issues.append(
                        ValidationIssue(
                            row.sample_id,
                            "invalid_label",
                            f"mask contains label(s) outside [0, {num_classes}): {invalid}",
                        )
                    )
    except OSError as exc:
        issues.append(ValidationIssue(row.sample_id, "unreadable_file", str(exc)))

    return issues
