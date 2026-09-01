# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the optional MLflow tracking adapter without contacting MLflow."""

from typing import Any

import pytest

from precisionai.agritune.tracking import mlflow_tracker
from precisionai.agritune.tracking.mlflow_tracker import MLflowTracker


class _FakeMLflow:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def set_tracking_uri(self, value: str) -> None:
        self.calls.append(("set_tracking_uri", value))

    def set_experiment(self, value: str) -> None:
        self.calls.append(("set_experiment", value))

    def start_run(self, *, run_name: str | None) -> object:
        self.calls.append(("start_run", run_name))
        return object()

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        self.calls.append(("log_metrics", (metrics, step)))

    def log_params(self, params: dict[str, Any]) -> None:
        self.calls.append(("log_params", params))

    def log_artifact(self, path: str) -> None:
        self.calls.append(("log_artifact", path))

    def end_run(self) -> None:
        self.calls.append(("end_run", None))


def test_raises_import_error_with_install_hint_when_mlflow_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mlflow_tracker, "_MLFLOW_AVAILABLE", False)
    with pytest.raises(ImportError, match=r"pip install pai-agritune\[tracking\]"):
        MLflowTracker()


def test_skips_optional_uri_and_experiment_when_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeMLflow()
    monkeypatch.setattr(mlflow_tracker, "_MLFLOW_AVAILABLE", True)
    monkeypatch.setattr(mlflow_tracker, "mlflow", fake)

    MLflowTracker()

    assert fake.calls == [("start_run", None)]


def test_delegates_the_full_tracker_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeMLflow()
    monkeypatch.setattr(mlflow_tracker, "_MLFLOW_AVAILABLE", True)
    monkeypatch.setattr(mlflow_tracker, "mlflow", fake)

    tracker = MLflowTracker(experiment_name="exp", tracking_uri="file:///tmp/mlruns", run_name="run")
    tracker.log_metrics({"loss": 1.0}, step=2)
    tracker.log_params({"lr": 0.1})
    tracker.log_artifact("model.ckpt")
    tracker.close()

    assert fake.calls == [
        ("set_tracking_uri", "file:///tmp/mlruns"),
        ("set_experiment", "exp"),
        ("start_run", "run"),
        ("log_metrics", ({"loss": 1.0}, 2)),
        ("log_params", {"lr": 0.1}),
        ("log_artifact", "model.ckpt"),
        ("end_run", None),
    ]
