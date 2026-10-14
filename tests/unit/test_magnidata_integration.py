# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the MagniData training adapter."""

from pathlib import Path
from typing import Any

import mlflow
import pytest

from precisionai.agritune.features.keys import EncoderFingerprint
from precisionai.agritune.features.store import DirectoryFeatureStore
from precisionai.agritune.integrations.magnidata import (
    MagniDataOptions,
    MagniDataTracker,
    run_magnidata_training,
)
from precisionai.agritune.services.training_service import TrainingRunConfig
from precisionai.agritune.tasks.segmentation.losses import SegmentationLossConfig
from precisionai.agritune.training.trainer import TrainerConfig
from tests.fixtures.manifest_factory import build_manifest


class _Client:
    def __init__(self) -> None:
        self.metrics: list[tuple[str, str, float, int]] = []
        self.requests = 0

    def log_metric(self, run_id: str, key: str, value: float, *, step: int) -> None:
        self.metrics.append((run_id, key, value, step))
        self.requests += 1

    def log_batch(self, run_id: str, *, metrics: list[Any]) -> None:
        self.metrics.extend((run_id, metric.key, metric.value, metric.step) for metric in metrics)
        self.requests += 1

    def get_run(self, run_id: str) -> Any:
        del run_id
        return type("Run", (), {"data": type("Data", (), {"params": {}})()})()

    def log_param(self, run_id: str, key: str, value: Any) -> None:
        del run_id, key, value

    def log_artifact(self, run_id: str, path: str) -> None:
        del run_id, path


def test_tracker_translates_agritune_metrics_to_magnidata_keys() -> None:
    client = _Client()
    tracker = MagniDataTracker(client, "run")

    tracker.log_metrics(
        {"train_loss": 0.8, "lr": 0.001, "val_loss": 0.7, "val_mean_iou": 0.5, "val_iou_class_1": 0.4},
        step=2,
    )
    tracker.flush_epoch(1)

    assert sorted(call[1] for call in client.metrics) == sorted(
        ["train_loss/loss", "Learning_rate/lr", "val_loss/loss", "val_metric/miou", "val_metric/iou_1"]
    )


def test_tracker_sends_nothing_until_the_epoch_ends_and_then_one_batch() -> None:
    client = _Client()
    tracker = MagniDataTracker(client, "run")

    for step, loss in enumerate([1.0, 0.5, 0.75], start=1):
        tracker.log_metrics({"train_loss": loss, "lr": 0.1 / step}, step=step)
    tracker.log_metrics({"val_loss": 0.6, "val_mean_iou": 0.3}, step=3)

    assert client.requests == 0
    tracker.flush_epoch(4)

    assert client.requests == 1
    logged = {key: (value, step) for _, key, value, step in client.metrics}
    assert logged["train_loss/loss"] == (pytest.approx(0.75), 4)
    assert logged["Learning_rate/lr"] == (pytest.approx(0.1 / 3), 4)
    assert logged["val_loss/loss"] == (0.6, 4)
    assert {step for *_, step in client.metrics} == {4}


def test_tracker_starts_each_epoch_afresh_and_skips_empty_flushes() -> None:
    client = _Client()
    tracker = MagniDataTracker(client, "run")
    tracker.log_metrics({"train_loss": 1.0}, step=1)
    tracker.flush_epoch(1)
    tracker.flush_epoch(2)
    tracker.log_metrics({"train_loss": 0.5}, step=2)
    tracker.flush_epoch(3)

    assert [(key, value, step) for _, key, value, step in client.metrics] == [
        ("train_loss/loss", 1.0, 1),
        ("train_loss/loss", 0.5, 3),
    ]
    assert client.requests == 2


def test_options_reject_invalid_score_interval() -> None:
    with pytest.raises(ValueError, match="score_every"):
        MagniDataOptions(score_every=0)


def test_options_reject_invalid_prediction_count() -> None:
    with pytest.raises(ValueError, match="top_k_predictions"):
        MagniDataOptions(top_k_predictions=-1)


def test_real_mlflow_run_receives_magnidata_metrics_and_artifacts(tmp_path: Path) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.create_experiment("magnidata/test", artifact_location=str(tmp_path / "artifacts"))
    created = client.create_run(
        experiment_id,
        tags={"magnidata.origin": "ui", "magnidata.dataset_source": "data_user/tiny.csv"},
    )
    run_id = created.info.run_id
    client.update_run(run_id, status="SCHEDULED")
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    manifest = build_manifest(dataset_root, image_size=(8, 8))
    config = TrainingRunConfig(
        manifest_path=str(manifest),
        feature_store_dir=str(tmp_path / "features"),
        run_root=str(tmp_path / "runs"),
        run_id=run_id,
        num_classes=3,
        encoder_fingerprint=EncoderFingerprint(model="fake-encoder", revision="fake-v1", preprocessing=""),
        feature_provider="online",
        batch_size=2,
        val_fraction=0.25,
        trainer=TrainerConfig(max_epochs=2),
        loss=SegmentationLossConfig(name="ce_dice"),
    )

    result = run_magnidata_training(
        config,
        store=DirectoryFeatureStore(tmp_path / "features"),
        run_id=run_id,
        tracking_uri=tracking_uri,
        options=MagniDataOptions(top_k_predictions=2),
        show_progress=False,
    )

    assert result is not None
    run = client.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.tags["magnidata.engine"] == "agritune"
    assert run.data.tags["magnidata.scored_epochs"] == "1,2"
    assert "train_loss/loss" in run.data.metrics
    assert "val_loss/loss" in run.data.metrics
    for key in run.data.metrics:
        history = client.get_metric_history(run_id, key)
        assert [point.step for point in history] == [1, 2], f"{key} must have exactly one point per epoch"
    artifact_paths = {item.path for item in client.list_artifacts(run_id, "magnidata")}
    assert "magnidata/split.csv" in artifact_paths
    assert "magnidata/per_sample" in artifact_paths
    per_sample = client.list_artifacts(run_id, "magnidata/per_sample")
    assert [item.path for item in per_sample] == [
        "magnidata/per_sample/epoch_0001.csv",
        "magnidata/per_sample/epoch_0002.csv",
    ]
