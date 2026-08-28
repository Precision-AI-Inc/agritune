# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.services.tracking_selection."""

from pathlib import Path

import pytest

from precisionai.agritune.logging import RunDirectory
from precisionai.agritune.services.tracking_selection import TrackingSelection, build_trackers
from precisionai.agritune.tracking.jsonl import JSONLTracker
from precisionai.agritune.tracking.multi import MultiTracker
from precisionai.agritune.tracking.null import NullTracker


def test_empty_backends_returns_null_tracker(tmp_path: Path) -> None:
    run_dir = RunDirectory(tmp_path, "run-1")
    tracker = build_trackers(TrackingSelection(backends=[]), run_dir=run_dir, run_id="run-1")
    assert isinstance(tracker, NullTracker)


def test_explicit_null_backend_returns_null_tracker(tmp_path: Path) -> None:
    run_dir = RunDirectory(tmp_path, "run-1")
    tracker = build_trackers(TrackingSelection(backends=["null"]), run_dir=run_dir, run_id="run-1")
    assert isinstance(tracker, NullTracker)


def test_single_backend_returns_that_tracker_directly(tmp_path: Path) -> None:
    run_dir = RunDirectory(tmp_path, "run-1")
    tracker = build_trackers(TrackingSelection(backends=["jsonl"]), run_dir=run_dir, run_id="run-1")
    assert isinstance(tracker, JSONLTracker)
    tracker.close()


def test_multiple_backends_returns_a_multi_tracker(tmp_path: Path) -> None:
    # tensorboard is an optional extra; skip rather than assume either way whether it's installed
    # in this environment, matching test_tensorboard_tracker.py's own convention.
    pytest.importorskip("tensorboard")
    run_dir = RunDirectory(tmp_path, "run-1")

    tracker = build_trackers(TrackingSelection(backends=["jsonl", "tensorboard"]), run_dir=run_dir, run_id="run-1")

    assert isinstance(tracker, MultiTracker)
    tracker.log_metrics({"loss": 1.0}, step=0)
    tracker.close()
    assert run_dir.metrics_path.is_file()


def test_jsonl_tracker_writes_under_the_run_directorys_metrics_path(tmp_path: Path) -> None:
    run_dir = RunDirectory(tmp_path, "run-1")
    tracker = build_trackers(TrackingSelection(backends=["jsonl"]), run_dir=run_dir, run_id="run-1")
    tracker.log_metrics({"loss": 1.0}, step=0)
    tracker.close()
    assert run_dir.metrics_path.is_file()


def test_unknown_backend_raises(tmp_path: Path) -> None:
    run_dir = RunDirectory(tmp_path, "run-1")
    with pytest.raises(ValueError, match="unknown tracking backend"):
        build_trackers(TrackingSelection(backends=["carrier-pigeon"]), run_dir=run_dir, run_id="run-1")


def test_neptune_without_a_project_raises(tmp_path: Path) -> None:
    run_dir = RunDirectory(tmp_path, "run-1")
    with pytest.raises(ValueError, match="neptune_project is required"):
        build_trackers(TrackingSelection(backends=["neptune"], neptune_project=None), run_dir=run_dir, run_id="run-1")


def test_mlflow_backend_selection_reaches_mlflow_trackers_constructor(tmp_path: Path) -> None:
    run_dir = RunDirectory(tmp_path, "run-1")
    with pytest.raises(ImportError, match=r"pip install pai-agritune\[tracking\]"):
        build_trackers(TrackingSelection(backends=["mlflow"]), run_dir=run_dir, run_id="run-1")


def test_wandb_backend_selection_reaches_wandb_trackers_constructor(tmp_path: Path) -> None:
    run_dir = RunDirectory(tmp_path, "run-1")
    with pytest.raises(ImportError, match=r"pip install pai-agritune\[tracking\]"):
        build_trackers(TrackingSelection(backends=["wandb"]), run_dir=run_dir, run_id="run-1")


def test_comet_backend_selection_reaches_comet_trackers_constructor(tmp_path: Path) -> None:
    run_dir = RunDirectory(tmp_path, "run-1")
    with pytest.raises(ImportError, match=r"pip install pai-agritune\[tracking\]"):
        build_trackers(TrackingSelection(backends=["comet"]), run_dir=run_dir, run_id="run-1")


def test_neptune_backend_selection_reaches_neptune_trackers_constructor(tmp_path: Path) -> None:
    run_dir = RunDirectory(tmp_path, "run-1")
    with pytest.raises(ImportError, match=r"pip install pai-agritune\[tracking\]"):
        build_trackers(
            TrackingSelection(backends=["neptune"], neptune_project="workspace/project"),
            run_dir=run_dir,
            run_id="run-1",
        )


def test_single_backend_never_wraps_in_multi_tracker(tmp_path: Path) -> None:
    run_dir = RunDirectory(tmp_path, "run-1")
    tracker = build_trackers(TrackingSelection(backends=["jsonl"]), run_dir=run_dir, run_id="run-1")
    assert not isinstance(tracker, MultiTracker)
    tracker.close()
