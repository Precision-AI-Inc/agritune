# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0
"""Dataset adapters, manifests, and split strategies."""

from precisionai.agritune.data.dataset import ManifestDataset, SegmentationDataset
from precisionai.agritune.data.manifest import ManifestRow, load_manifest, parse_manifest_rows
from precisionai.agritune.data.split import (
    SplitAssignment,
    detect_group_leakage,
    grouped_split,
    random_split,
)
from precisionai.agritune.data.validation import ValidationIssue, ValidationReport, validate_manifest

__all__ = [
    "ManifestDataset",
    "ManifestRow",
    "SegmentationDataset",
    "SplitAssignment",
    "ValidationIssue",
    "ValidationReport",
    "detect_group_leakage",
    "grouped_split",
    "load_manifest",
    "parse_manifest_rows",
    "random_split",
    "validate_manifest",
]
