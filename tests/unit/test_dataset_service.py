# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.services.dataset_service."""

from pathlib import Path

import pytest

from precisionai.agritune.data.manifest import parse_manifest_rows
from precisionai.agritune.services.dataset_service import (
    inspect_dataset,
    render_manifest_template,
    validate_dataset,
    write_manifest_template,
)
from tests.fixtures.manifest_factory import build_manifest


def test_validate_dataset_reports_no_issues_for_a_clean_manifest(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path)
    report = validate_dataset(str(manifest_path))
    assert report.is_valid


def test_validate_dataset_detects_missing_image(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path, omit_image_for="sample-1")
    report = validate_dataset(str(manifest_path))
    assert not report.is_valid


def test_validate_dataset_checks_num_classes(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path)  # sample-2's mask is filled with value 2
    report = validate_dataset(str(manifest_path), num_classes=2)
    assert any(issue.category == "invalid_label" for issue in report.issues)


def test_inspect_dataset_reports_sample_count_and_columns(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path)
    result = inspect_dataset(str(manifest_path))
    assert result["num_samples"] == 4
    assert result["metadata_columns"] == ["field_id"]


def test_inspect_dataset_reports_class_pixel_counts(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path, image_size=(2, 2))
    result = inspect_dataset(str(manifest_path))
    # Default fixture rows have mask fill values 0, 1, 2, 0 over 2x2=4-pixel masks.
    assert result["class_pixel_counts"] == {0: 8, 1: 4, 2: 4}


def test_inspect_dataset_skips_unreadable_masks(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path, omit_mask_for="sample-1")
    result = inspect_dataset(str(manifest_path))
    assert result["num_samples"] == 4  # still counts the row, just skips its mask stats


def test_render_manifest_template_is_a_parseable_manifest_with_placeholder_rows(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    path.write_text(render_manifest_template(), encoding="utf-8")

    rows = parse_manifest_rows(path)

    assert len(rows) >= 1
    for row in rows:
        assert row.sample_id
        assert row.image_path
        assert row.mask_path


def test_write_manifest_template_creates_the_file(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "manifest.csv"
    result = write_manifest_template(output)
    assert result == output
    assert output.read_text(encoding="utf-8") == render_manifest_template()


def test_write_manifest_template_refuses_to_overwrite_by_default(tmp_path: Path) -> None:
    output = tmp_path / "manifest.csv"
    output.write_text("existing content", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        write_manifest_template(output)
    assert output.read_text(encoding="utf-8") == "existing content"


def test_write_manifest_template_overwrites_when_forced(tmp_path: Path) -> None:
    output = tmp_path / "manifest.csv"
    output.write_text("existing content", encoding="utf-8")
    write_manifest_template(output, force=True)
    assert output.read_text(encoding="utf-8") == render_manifest_template()
