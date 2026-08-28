# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.tracking.multi.MultiTracker."""

from typing import Any

import pytest

from precisionai.agritune.schemas.protocols import Tracker
from precisionai.agritune.tracking.multi import MultiTracker


class _RecordingTracker:
    def __init__(self, *, raise_on_close: bool = False) -> None:
        self.metrics_calls: list[tuple[dict[str, float], int]] = []
        self.params_calls: list[dict[str, Any]] = []
        self.artifact_calls: list[str] = []
        self.closed = False
        self._raise_on_close = raise_on_close

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        self.metrics_calls.append((metrics, step))

    def log_params(self, params: dict[str, Any]) -> None:
        self.params_calls.append(params)

    def log_artifact(self, path: str) -> None:
        self.artifact_calls.append(path)

    def close(self) -> None:
        self.closed = True
        if self._raise_on_close:
            raise RuntimeError("boom")


def test_log_metrics_forwards_to_every_tracker() -> None:
    first, second = _RecordingTracker(), _RecordingTracker()
    MultiTracker([first, second]).log_metrics({"loss": 1.0}, step=1)
    assert first.metrics_calls == [({"loss": 1.0}, 1)]
    assert second.metrics_calls == [({"loss": 1.0}, 1)]


def test_log_params_forwards_to_every_tracker() -> None:
    first, second = _RecordingTracker(), _RecordingTracker()
    MultiTracker([first, second]).log_params({"lr": 0.1})
    assert first.params_calls == [{"lr": 0.1}]
    assert second.params_calls == [{"lr": 0.1}]


def test_log_artifact_forwards_to_every_tracker() -> None:
    first, second = _RecordingTracker(), _RecordingTracker()
    MultiTracker([first, second]).log_artifact("best.ckpt")
    assert first.artifact_calls == ["best.ckpt"]
    assert second.artifact_calls == ["best.ckpt"]


def test_close_closes_every_tracker() -> None:
    first, second = _RecordingTracker(), _RecordingTracker()
    MultiTracker([first, second]).close()
    assert first.closed
    assert second.closed


def test_close_still_closes_remaining_trackers_when_one_raises() -> None:
    first = _RecordingTracker(raise_on_close=True)
    second = _RecordingTracker()
    with pytest.raises(RuntimeError, match="boom"):
        MultiTracker([first, second]).close()
    assert first.closed
    assert second.closed


def test_multi_tracker_satisfies_protocol() -> None:
    assert isinstance(MultiTracker([]), Tracker)
