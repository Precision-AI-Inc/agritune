# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.logging.progress and progress gating."""

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import TypeVar

import pytest
from fastapi.testclient import TestClient

from precisionai.agritune.api import create_app
from precisionai.agritune.cli import handlers
from precisionai.agritune.cli.main import main
from precisionai.agritune.logging import progress_iter
from tests.fixtures.manifest_factory import build_manifest

T = TypeVar("T")


def test_progress_iter_yields_items_unchanged_when_enabled() -> None:
    assert list(progress_iter([1, 2, 3], desc="items")) == [1, 2, 3]


def test_progress_iter_yields_items_unchanged_when_disabled() -> None:
    assert list(progress_iter([1, 2, 3], disable=True)) == [1, 2, 3]


def test_progress_iter_handles_iterables_without_len() -> None:
    assert list(progress_iter(iter([1, 2, 3]))) == [1, 2, 3]


def test_progress_iter_handles_empty_iterable() -> None:
    assert list(progress_iter([])) == []


def test_progress_iter_forwards_description_unit_disable_and_leave(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_tqdm(iterable: Iterable[T], **kwargs: object) -> Iterable[T]:
        captured.update(kwargs)
        return iterable

    monkeypatch.setattr("precisionai.agritune.logging.progress.tqdm", fake_tqdm)
    assert list(progress_iter(["a"], desc="epoch 0", unit="batch", disable=False)) == ["a"]
    assert captured["desc"] == "epoch 0"
    assert captured["unit"] == "batch"
    assert captured["disable"] is False
    assert captured["leave"] is False


def test_progress_iter_writes_description_to_stderr_when_enabled(capsys: pytest.CaptureFixture[str]) -> None:
    list(progress_iter(range(2), desc="items", unit="sample", disable=False))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "items" in captured.err


def test_progress_iter_writes_nothing_when_disabled(capsys: pytest.CaptureFixture[str]) -> None:
    list(progress_iter(range(2), desc="items", unit="sample", disable=True))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_cli_train_enables_progress(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, bool] = {}

    def fake_run_training(_config: object, *, store: object, show_progress: bool = False) -> SimpleNamespace:
        seen["show_progress"] = show_progress
        return SimpleNamespace(
            run_directory=SimpleNamespace(path=tmp_path / "run"),
            final_train_state={"epoch": 1},
            train_metrics={},
            val_metrics={},
        )

    monkeypatch.setattr(
        handlers,
        "load_training_run_config",
        lambda _config, _overrides: SimpleNamespace(feature_store_dir=str(tmp_path)),
    )
    monkeypatch.setattr(handlers, "DirectoryFeatureStore", lambda _path: object())
    monkeypatch.setattr(handlers, "run_training", fake_run_training)
    exit_code = handlers.train(argparse.Namespace(config="train.yaml", overrides=[]))
    assert exit_code == 0
    assert seen["show_progress"] is True


def test_api_train_evaluate_and_predict_stay_headless(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AGRITUNE_API_ROOT", str(tmp_path))
    seen: dict[str, bool] = {}

    def fake_run_training(_config: object, *, store: object, show_progress: bool = False) -> SimpleNamespace:
        seen["train"] = show_progress
        return SimpleNamespace(
            run_directory=SimpleNamespace(path=str(tmp_path / "run")),
            final_train_state={"epoch": 0, "global_optimizer_step": 0, "best_metric": None},
            train_metrics={},
            val_metrics={},
        )

    def fake_run_evaluation(_config: object, *, store: object, show_progress: bool = False) -> dict[str, float]:
        seen["evaluate"] = show_progress
        return {"mean_iou": 0.0}

    def fake_run_prediction(_config: object, *, store: object, show_progress: bool = False) -> list[str]:
        seen["predict"] = show_progress
        return []

    monkeypatch.setattr(
        "precisionai.agritune.api.routes.train.load_training_run_config",
        lambda _path, _overrides: SimpleNamespace(feature_store_dir=str(tmp_path)),
    )
    monkeypatch.setattr("precisionai.agritune.api.routes.train.DirectoryFeatureStore", lambda _path: object())
    monkeypatch.setattr("precisionai.agritune.api.routes.train.run_training", fake_run_training)
    monkeypatch.setattr("precisionai.agritune.api.routes.evaluate.DirectoryFeatureStore", lambda _path: object())
    monkeypatch.setattr("precisionai.agritune.api.routes.evaluate.run_evaluation", fake_run_evaluation)
    monkeypatch.setattr("precisionai.agritune.api.routes.predict.DirectoryFeatureStore", lambda _path: object())
    monkeypatch.setattr("precisionai.agritune.api.routes.predict.run_prediction", fake_run_prediction)

    client = TestClient(create_app())
    train_response = client.post("/train", json={"config_path": str(tmp_path / "unused.yaml")})
    evaluate_response = client.post(
        "/evaluate",
        json={
            "manifest_path": str(tmp_path / "manifest.csv"),
            "store": str(tmp_path / "features"),
            "checkpoint_path": str(tmp_path / "last.ckpt"),
            "num_classes": 2,
        },
    )
    predict_response = client.post(
        "/predict",
        json={
            "manifest_path": str(tmp_path / "manifest.csv"),
            "store": str(tmp_path / "features"),
            "checkpoint_path": str(tmp_path / "last.ckpt"),
            "output_dir": str(tmp_path / "out"),
            "num_classes": 2,
        },
    )
    assert train_response.status_code == 200
    assert evaluate_response.status_code == 200
    assert predict_response.status_code == 200
    assert seen == {"train": False, "evaluate": False, "predict": False}


def test_dataset_inspect_keeps_stdout_machine_readable_when_debug_logging(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = build_manifest(tmp_path)
    exit_code = main(["--log-level", "DEBUG", "dataset", "inspect", "--manifest", str(manifest_path)])
    assert exit_code == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["num_samples"] == 4
    assert "dispatching dataset inspect" in captured.err
