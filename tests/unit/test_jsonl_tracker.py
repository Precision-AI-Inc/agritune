# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.tracking.jsonl.JSONLTracker."""

import json
from pathlib import Path

from precisionai.agritune.schemas.protocols import Tracker
from precisionai.agritune.tracking.jsonl import JSONLTracker


def _read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_log_metrics_writes_one_json_line(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    tracker = JSONLTracker(path)
    tracker.log_metrics({"loss": 0.5}, step=3)
    tracker.close()

    records = _read_records(path)
    assert len(records) == 1
    assert records[0]["type"] == "metrics"
    assert records[0]["step"] == 3
    assert records[0]["data"] == {"loss": 0.5}


def test_log_params_and_artifact_write_distinct_records(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    tracker = JSONLTracker(path)
    tracker.log_params({"lr": 0.1})
    tracker.log_artifact("best.ckpt")
    tracker.close()

    records = _read_records(path)
    assert records[0]["type"] == "params"
    assert records[0]["data"] == {"lr": 0.1}
    assert records[1]["type"] == "artifact"
    assert records[1]["path"] == "best.ckpt"


def test_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "metrics.jsonl"
    tracker = JSONLTracker(path)
    tracker.close()
    assert path.parent.is_dir()


def test_appends_across_tracker_instances(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    first = JSONLTracker(path)
    first.log_metrics({"loss": 1.0}, step=0)
    first.close()

    second = JSONLTracker(path)
    second.log_metrics({"loss": 0.5}, step=1)
    second.close()

    records = _read_records(path)
    assert len(records) == 2


def test_jsonl_tracker_satisfies_protocol(tmp_path: Path) -> None:
    tracker = JSONLTracker(tmp_path / "metrics.jsonl")
    assert isinstance(tracker, Tracker)
    tracker.close()
