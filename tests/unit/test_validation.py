# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.data.validation."""

from pathlib import Path

from precisionai.agritune.data.validation import validate_manifest
from tests.fixtures.manifest_factory import build_manifest


def test_valid_manifest_has_no_issues(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path)
    report = validate_manifest(manifest_path)
    assert report.is_valid


def test_missing_image_is_reported(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path, omit_image_for="sample-1")
    report = validate_manifest(manifest_path)
    assert not report.is_valid
    assert any(issue.category == "missing_image" and issue.sample_id == "sample-1" for issue in report.issues)


def test_missing_mask_is_reported(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path, omit_mask_for="sample-2")
    report = validate_manifest(manifest_path)
    assert any(issue.category == "missing_mask" and issue.sample_id == "sample-2" for issue in report.issues)


def test_dimension_mismatch_is_reported(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path, mismatched_mask_size_for="sample-0")
    report = validate_manifest(manifest_path)
    assert any(issue.category == "dimension_mismatch" and issue.sample_id == "sample-0" for issue in report.issues)


def test_dimension_check_can_be_disabled(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path, mismatched_mask_size_for="sample-0")
    report = validate_manifest(manifest_path, check_image_mask_dimensions=False)
    assert not any(issue.category == "dimension_mismatch" for issue in report.issues)


def test_duplicate_id_is_reported(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    (tmp_path / "a.png").write_bytes(b"")
    manifest_path.write_text(
        "sample_id,image_path,mask_path\ns1,a.png,a.png\ns1,a.png,a.png\n",
        encoding="utf-8",
    )
    report = validate_manifest(manifest_path, check_image_mask_dimensions=False)
    assert any(issue.category == "duplicate_id" for issue in report.issues)


def test_invalid_label_is_reported_when_num_classes_given(tmp_path: Path) -> None:
    # sample-2's mask is filled with value 2; restrict to 2 classes (valid labels 0, 1).
    manifest_path = build_manifest(tmp_path)
    report = validate_manifest(manifest_path, num_classes=2)
    assert any(issue.category == "invalid_label" and issue.sample_id == "sample-2" for issue in report.issues)


def test_invalid_label_respects_ignore_index(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path)
    report = validate_manifest(manifest_path, num_classes=2, ignore_index=2)
    assert not any(issue.category == "invalid_label" for issue in report.issues)


def test_issues_by_category_groups_correctly(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path, omit_image_for="sample-1", omit_mask_for="sample-2")
    report = validate_manifest(manifest_path)
    grouped = report.issues_by_category()
    assert "missing_image" in grouped
    assert "missing_mask" in grouped
