# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the optional Weights & Biases adapter without network access."""

from typing import Any

import pytest

from precisionai.agritune.tracking import wandb_tracker
from precisionai.agritune.tracking.wandb_tracker import WandBTracker


class _FakeConfig:
    def __init__(self, calls: list[tuple[str, Any]]) -> None:
        self._calls = calls

    def update(self, params: dict[str, Any]) -> None:
        self._calls.append(("config.update", params))


class _FakeRun:
    def __init__(self, calls: list[tuple[str, Any]]) -> None:
        self._calls = calls
        self.config = _FakeConfig(calls)

    def log(self, metrics: dict[str, float], *, step: int) -> None:
        self._calls.append(("log", (metrics, step)))

    def log_artifact(self, path: str) -> None:
        self._calls.append(("log_artifact", path))

    def finish(self) -> None:
        self._calls.append(("finish", None))


class _FakeWandB:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def init(self, *, project: str, name: str | None) -> _FakeRun:
        self.calls.append(("init", (project, name)))
        return _FakeRun(self.calls)


def test_raises_import_error_with_install_hint_when_wandb_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wandb_tracker, "_WANDB_AVAILABLE", False)
    with pytest.raises(ImportError, match=r"pip install pai-agritune\[tracking\]"):
        WandBTracker(project="agritune-test")


def test_delegates_the_full_tracker_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeWandB()
    monkeypatch.setattr(wandb_tracker, "_WANDB_AVAILABLE", True)
    monkeypatch.setattr(wandb_tracker, "wandb", fake)

    tracker = WandBTracker(project="project", run_name="run")
    tracker.log_metrics({"loss": 1.0}, step=2)
    tracker.log_params({"lr": 0.1})
    tracker.log_artifact("model.ckpt")
    tracker.close()

    assert fake.calls == [
        ("init", ("project", "run")),
        ("log", ({"loss": 1.0}, 2)),
        ("config.update", {"lr": 0.1}),
        ("log_artifact", "model.ckpt"),
        ("finish", None),
    ]
