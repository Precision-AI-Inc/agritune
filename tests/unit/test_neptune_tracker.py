# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the optional Neptune adapter without network access."""

from typing import Any

import pytest

from precisionai.agritune.tracking import neptune_tracker
from precisionai.agritune.tracking.neptune_tracker import NeptuneTracker


class _FakeField:
    def __init__(self, name: str, calls: list[tuple[str, Any]]) -> None:
        self._name = name
        self._calls = calls

    def append(self, value: float, *, step: int) -> None:
        self._calls.append(("append", (self._name, value, step)))

    def upload(self, path: str) -> None:
        self._calls.append(("upload", (self._name, path)))


class _FakeRun:
    def __init__(self, calls: list[tuple[str, Any]]) -> None:
        self._calls = calls

    def __getitem__(self, name: str) -> _FakeField:
        return _FakeField(name, self._calls)

    def __setitem__(self, name: str, value: Any) -> None:
        self._calls.append(("setitem", (name, value)))

    def stop(self) -> None:
        self._calls.append(("stop", None))


class _FakeNeptune:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def init_run(self, *, project: str, api_token: str | None, name: str | None) -> _FakeRun:
        self.calls.append(("init_run", (project, api_token, name)))
        return _FakeRun(self.calls)


def test_raises_import_error_with_install_hint_when_neptune_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(neptune_tracker, "_NEPTUNE_AVAILABLE", False)
    with pytest.raises(ImportError, match=r"pip install pai-agritune\[tracking\]"):
        NeptuneTracker(project="workspace/agritune-test")


def test_delegates_the_full_tracker_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeNeptune()
    monkeypatch.setattr(neptune_tracker, "_NEPTUNE_AVAILABLE", True)
    monkeypatch.setattr(neptune_tracker, "neptune", fake)

    tracker = NeptuneTracker(project="workspace/project", api_token="token", run_name="run")
    tracker.log_metrics({"loss": 1.0, "iou": 0.5}, step=2)
    tracker.log_params({"lr": 0.1})
    tracker.log_artifact("dir/model.ckpt")
    tracker.close()

    assert fake.calls == [
        ("init_run", ("workspace/project", "token", "run")),
        ("append", ("metrics/loss", 1.0, 2)),
        ("append", ("metrics/iou", 0.5, 2)),
        ("setitem", ("parameters", {"lr": 0.1})),
        ("upload", ("artifacts/model.ckpt", "dir/model.ckpt")),
        ("stop", None),
    ]
