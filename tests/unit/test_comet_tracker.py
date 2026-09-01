# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the optional Comet adapter without network access."""

from typing import Any

import pytest

from precisionai.agritune.tracking import comet_tracker
from precisionai.agritune.tracking.comet_tracker import CometTracker


class _FakeExperiment:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        self.calls.append(("log_metrics", (metrics, step)))

    def log_parameters(self, params: dict[str, Any]) -> None:
        self.calls.append(("log_parameters", params))

    def log_asset(self, path: str) -> None:
        self.calls.append(("log_asset", path))

    def end(self) -> None:
        self.calls.append(("end", None))


class _FakeComet:
    def __init__(self) -> None:
        self.constructor_args: tuple[str, str | None, str | None] | None = None
        self.experiment = _FakeExperiment()

    def __getattr__(self, name: str) -> Any:
        if name == "Experiment":
            return self._build_experiment
        raise AttributeError(name)

    def _build_experiment(self, *, project_name: str, api_key: str | None, workspace: str | None) -> _FakeExperiment:
        self.constructor_args = (project_name, api_key, workspace)
        return self.experiment


def test_importing_module_does_not_import_comet_ml() -> None:
    """Regression test: comet_ml auto-instruments other frameworks (e.g. mlflow) merely by being
    imported, so importing this module — which happens unconditionally via
    tracking_selection.py's backend registry, regardless of which backend is selected — must
    never trigger a real ``import comet_ml`` as a side effect. Only constructing a CometTracker
    may do that.
    """
    assert comet_tracker._comet_import_attempted is False
    assert comet_tracker.comet_ml is None


def test_raises_import_error_with_install_hint_when_comet_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(comet_tracker, "_comet_import_attempted", True)
    monkeypatch.setattr(comet_tracker, "_COMET_AVAILABLE", False)
    with pytest.raises(ImportError, match=r"pip install pai-agritune\[tracking\]"):
        CometTracker(project_name="agritune-test")


def test_delegates_the_full_tracker_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeComet()
    monkeypatch.setattr(comet_tracker, "_comet_import_attempted", True)
    monkeypatch.setattr(comet_tracker, "_COMET_AVAILABLE", True)
    monkeypatch.setattr(comet_tracker, "comet_ml", fake)

    tracker = CometTracker(project_name="project", api_key="token", workspace="workspace")
    tracker.log_metrics({"loss": 1.0}, step=2)
    tracker.log_params({"lr": 0.1})
    tracker.log_artifact("model.ckpt")
    tracker.close()

    assert fake.constructor_args == ("project", "token", "workspace")
    assert fake.experiment.calls == [
        ("log_metrics", ({"loss": 1.0}, 2)),
        ("log_parameters", {"lr": 0.1}),
        ("log_asset", "model.ckpt"),
        ("end", None),
    ]
