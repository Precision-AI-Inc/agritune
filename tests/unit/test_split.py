# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.data.split."""

from pathlib import Path

import pytest

from precisionai.agritune.data.manifest import ManifestRow, parse_manifest_rows
from precisionai.agritune.data.split import (
    SplitAssignment,
    detect_group_leakage,
    grouped_split,
    random_split,
    temporal_split,
)
from tests.fixtures.manifest_factory import build_manifest


def _rows(tmp_path: Path) -> list[ManifestRow]:
    return parse_manifest_rows(build_manifest(tmp_path))


def _dated_row(sample_id: str, capture_date: str) -> ManifestRow:
    return ManifestRow(
        sample_id=sample_id,
        image_path=f"{sample_id}.png",
        mask_path=f"{sample_id}_mask.png",
        metadata={"capture_date": capture_date},
    )


def test_random_split_assigns_every_sample_exactly_once(tmp_path: Path) -> None:
    split = random_split(_rows(tmp_path), seed=0)
    all_ids = split.train + split.val + split.test
    assert sorted(all_ids) == ["sample-0", "sample-1", "sample-2", "sample-3"]
    assert len(set(all_ids)) == len(all_ids)


def test_random_split_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    first = random_split(rows, seed=42)
    second = random_split(rows, seed=42)
    assert first == second


def test_random_split_invalid_fraction_sum_raises(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    with pytest.raises(ValueError, match="train_fraction \\+ val_fraction"):
        random_split(rows, train_fraction=0.9, val_fraction=0.2)


def test_random_split_train_fraction_out_of_range_raises(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    with pytest.raises(ValueError, match="train_fraction must be in"):
        random_split(rows, train_fraction=1.5, val_fraction=0.1)


def test_random_split_val_fraction_out_of_range_raises(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    with pytest.raises(ValueError, match="val_fraction must be in"):
        random_split(rows, train_fraction=0.5, val_fraction=-0.1)


def test_grouped_split_keeps_group_together(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    split = grouped_split(rows, group_by="field_id", seed=1)
    sample_to_group = {row.sample_id: row.metadata["field_id"] for row in rows}
    for subset in (split.train, split.val, split.test):
        groups_in_subset = {sample_to_group[sample_id] for sample_id in subset}
        assert len(detect_group_leakage(rows, split, group_by="field_id")) == 0
        del groups_in_subset  # only used for readability above


def test_grouped_split_can_assign_groups_to_val(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    split = grouped_split(rows, group_by="field_id", train_fraction=1e-6, val_fraction=0.999997, seed=1)
    assert split.val
    assert len(split.train) + len(split.val) + len(split.test) == len(rows)


def test_grouped_split_can_assign_groups_to_test(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    split = grouped_split(rows, group_by="field_id", train_fraction=1e-6, val_fraction=1e-6, seed=1)
    assert split.test
    assert len(split.train) + len(split.val) + len(split.test) == len(rows)


def test_grouped_split_missing_group_metadata_raises(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    rows[0].metadata = {}
    with pytest.raises(ValueError, match="missing grouping metadata"):
        grouped_split(rows, group_by="field_id")


def test_detect_group_leakage_flags_split_groups(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    # sample-0 and sample-1 share field-a; force them into different splits manually.
    leaking_split = SplitAssignment(train=["sample-0"], val=["sample-1"], test=["sample-2", "sample-3"])
    leaked = detect_group_leakage(rows, leaking_split, group_by="field_id")
    assert leaked == ["field-a"]


def test_detect_group_leakage_empty_when_consistent(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    clean_split = SplitAssignment(train=["sample-0", "sample-1"], val=[], test=["sample-2", "sample-3"])
    assert detect_group_leakage(rows, clean_split, group_by="field_id") == []


def test_detect_group_leakage_ignores_samples_absent_from_the_split(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    partial_split = SplitAssignment(train=["sample-0"], val=[], test=["sample-2", "sample-3"])
    assert detect_group_leakage(rows, partial_split, group_by="field_id") == []


def test_temporal_split_assigns_earliest_to_train_and_latest_to_test() -> None:
    rows = [
        _dated_row("a", "2024-01-01"),
        _dated_row("b", "2024-02-01"),
        _dated_row("c", "2024-03-01"),
        _dated_row("d", "2024-04-01"),
        _dated_row("e", "2024-05-01"),
    ]
    split = temporal_split(rows, train_fraction=0.6, val_fraction=0.2)
    assert split.train == ["a", "b", "c"]
    assert split.val == ["d"]
    assert split.test == ["e"]


def test_temporal_split_assigns_every_sample_exactly_once() -> None:
    rows = [_dated_row(f"s{i}", f"2024-01-{i + 1:02d}") for i in range(6)]
    split = temporal_split(rows, train_fraction=0.5, val_fraction=0.3)
    all_ids = split.train + split.val + split.test
    assert sorted(all_ids) == sorted(row.sample_id for row in rows)
    assert len(set(all_ids)) == len(all_ids)


def test_temporal_split_is_order_independent_in_the_input() -> None:
    forward = [
        _dated_row("a", "2024-01-01"),
        _dated_row("b", "2024-02-01"),
        _dated_row("c", "2024-03-01"),
    ]
    shuffled = [forward[2], forward[0], forward[1]]
    assert temporal_split(forward, train_fraction=0.6, val_fraction=0.2) == temporal_split(
        shuffled, train_fraction=0.6, val_fraction=0.2
    )


def test_temporal_split_missing_date_metadata_raises() -> None:
    rows = [_dated_row("a", "2024-01-01"), ManifestRow(sample_id="b", image_path="b.png", mask_path="b.png")]
    with pytest.raises(ValueError, match="missing 'capture_date' metadata"):
        temporal_split(rows)


def test_temporal_split_invalid_fraction_sum_raises() -> None:
    rows = [_dated_row("a", "2024-01-01")]
    with pytest.raises(ValueError, match="train_fraction \\+ val_fraction"):
        temporal_split(rows, train_fraction=0.9, val_fraction=0.2)
