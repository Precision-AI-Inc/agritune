# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.data.manifest."""

from pathlib import Path

import pytest

from precisionai.agritune.data.manifest import load_manifest, parse_manifest_rows
from tests.fixtures.manifest_factory import build_manifest


def test_parse_manifest_rows_reads_required_columns(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path)
    rows = parse_manifest_rows(manifest_path)
    assert len(rows) == 4
    assert rows[0].sample_id == "sample-0"
    assert rows[0].image_path == "sample-0_image.png"


def test_parse_manifest_rows_captures_extra_columns_as_metadata(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path)
    rows = parse_manifest_rows(manifest_path)
    assert rows[0].metadata == {"field_id": "field-a"}


def test_parse_manifest_rows_missing_required_column_raises(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("sample_id,image_path\ns1,img.png\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required column"):
        parse_manifest_rows(manifest_path)


def test_parse_manifest_rows_no_header_raises(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no header row"):
        parse_manifest_rows(manifest_path)


def test_load_manifest_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "sample_id,image_path,mask_path\ns1,a.png,a_mask.png\ns1,b.png,b_mask.png\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate sample_id"):
        load_manifest(manifest_path)


def test_load_manifest_accepts_unique_ids(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path)
    rows = load_manifest(manifest_path)
    assert {row.sample_id for row in rows} == {"sample-0", "sample-1", "sample-2", "sample-3"}
