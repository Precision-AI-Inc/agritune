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

    def log_metric(self, run_id: str, key: str, value: float, *, step: int) -> None:
        self.metrics.append((run_id, key, value, step))

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

    assert [call[1] for call in client.metrics] == [
        "train_loss/loss",
        "Learning_rate/lr",
        "val_loss/loss",
        "val_metric/miou",
        "val_metric/iou_1",
    ]


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
        trainer=TrainerConfig(max_epochs=1),
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
    assert run.data.tags["magnidata.scored_epochs"] == "1"
    assert "train_loss/loss" in run.data.metrics
    assert "val_loss/loss" in run.data.metrics
    artifact_paths = {item.path for item in client.list_artifacts(run_id, "magnidata")}
    assert "magnidata/split.csv" in artifact_paths
    assert "magnidata/per_sample" in artifact_paths
    per_sample = client.list_artifacts(run_id, "magnidata/per_sample")
    assert [item.path for item in per_sample] == ["magnidata/per_sample/epoch_0001.csv"]
