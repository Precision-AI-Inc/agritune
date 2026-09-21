# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for /train, /evaluate, /predict, and /encoder/benchmark, via
``fastapi.testclient.TestClient`` — mirrors ``tests/unit/test_cli_handlers.py``'s coverage of the
same underlying services through the other entry point.
"""

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from precisionai.agritune.api import create_app
from tests.fixtures.manifest_factory import build_manifest


@pytest.fixture(autouse=True)
def _api_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Scope the API's path-containment root to this test's ``tmp_path``."""
    monkeypatch.setenv("AGRITUNE_API_ROOT", str(tmp_path))


_ROWS = [
    ("sample-0", "field-a", 0),
    ("sample-1", "field-a", 1),
    ("sample-2", "field-b", 1),
    ("sample-3", "field-b", 0),
    ("sample-4", "field-c", 1),
    ("sample-5", "field-c", 0),
]


def _client() -> TestClient:
    return TestClient(create_app())


def _train_a_checkpoint(client: TestClient, tmp_path: Path, *, run_id: str = "api-run") -> tuple[Path, Path, Path]:
    """Build features then train via the API; return (manifest, store, checkpoint) paths."""
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store_path = tmp_path / "features"
    build_response = client.post(
        "/features/build", json={"manifest_path": str(manifest_path), "store": str(store_path)}
    )
    assert build_response.status_code == 200

    config = {
        "manifest_path": str(manifest_path),
        "feature_store_dir": str(store_path),
        "run_root": str(tmp_path / "runs"),
        "run_id": run_id,
        "num_classes": 2,
        "encoder_fingerprint": {"model": "fake-encoder", "revision": "fake-v1", "preprocessing": ""},
        "batch_size": 2,
        "val_fraction": 0.34,
        "optimizer": {"name": "adamw", "lr": 0.05},
        "trainer": {"max_epochs": 1},
    }
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(config))

    train_response = client.post("/train", json={"config_path": str(config_path)})
    assert train_response.status_code == 200

    checkpoint_path = tmp_path / "runs" / run_id / "checkpoints" / "last.ckpt"
    return manifest_path, store_path, checkpoint_path


def test_train_with_feature_augmentation_config_runs_full_pipeline(tmp_path: Path) -> None:
    client = _client()
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store_path = tmp_path / "features"
    build_response = client.post(
        "/features/build", json={"manifest_path": str(manifest_path), "store": str(store_path)}
    )
    assert build_response.status_code == 200

    config = {
        "manifest_path": str(manifest_path),
        "feature_store_dir": str(store_path),
        "run_root": str(tmp_path / "runs"),
        "run_id": "api-feature-aug-run",
        "num_classes": 2,
        "encoder_fingerprint": {"model": "fake-encoder", "revision": "fake-v1", "preprocessing": ""},
        "batch_size": 2,
        "val_fraction": 0.34,
        "optimizer": {"name": "adamw", "lr": 0.05},
        "trainer": {"max_epochs": 1},
        "feature_augmentation": {"patch_dropout_probability": 0.1, "gaussian_noise_std": 0.05},
    }
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(config))

    response = client.post("/train", json={"config_path": str(config_path)})

    assert response.status_code == 200
    assert response.json()["final_epoch"] == 1
    assert "mean_iou" in response.json()["train_metrics"]


def test_features_build_with_offline_augmentation_computes_the_augmented_variant_too(tmp_path: Path) -> None:
    client = _client()
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store_path = tmp_path / "features"
    augmentation_config_path = tmp_path / "augmentation.yaml"
    augmentation_config_path.write_text(
        yaml.safe_dump({"mode": "offline", "variant": 0, "geometric": {"horizontal_flip_probability": 0.5}})
    )

    response = client.post(
        "/features/build",
        json={
            "manifest_path": str(manifest_path),
            "store": str(store_path),
            "augmentation_config_path": str(augmentation_config_path),
            "seed": 0,
        },
    )

    assert response.status_code == 200
    assert response.json()["computed"] == 12  # 6 samples x (unaugmented + one offline variant)


def test_train_runs_full_pipeline_and_reports_metrics(tmp_path: Path) -> None:
    client = _client()
    _, _, checkpoint_path = _train_a_checkpoint(client, tmp_path)
    assert checkpoint_path.is_file()


def test_train_applies_dotlist_overrides(tmp_path: Path) -> None:
    client = _client()
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store_path = tmp_path / "features"
    client.post("/features/build", json={"manifest_path": str(manifest_path), "store": str(store_path)})

    config = {
        "manifest_path": str(manifest_path),
        "feature_store_dir": str(store_path),
        "run_root": str(tmp_path / "runs"),
        "run_id": "override-run",
        "num_classes": 2,
        "encoder_fingerprint": {"model": "fake-encoder", "revision": "fake-v1", "preprocessing": ""},
        "batch_size": 2,
        "val_fraction": 0.34,
        "trainer": {"max_epochs": 1},
    }
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(config))

    response = client.post("/train", json={"config_path": str(config_path), "overrides": ["trainer.max_epochs=3"]})

    assert response.status_code == 200
    assert response.json()["final_epoch"] == 3
    assert "mean_iou" in response.json()["train_metrics"]


def test_evaluate_reports_metrics(tmp_path: Path) -> None:
    client = _client()
    manifest_path, store_path, checkpoint_path = _train_a_checkpoint(client, tmp_path)

    response = client.post(
        "/evaluate",
        json={
            "manifest_path": str(manifest_path),
            "store": str(store_path),
            "checkpoint_path": str(checkpoint_path),
            "num_classes": 2,
        },
    )

    assert response.status_code == 200
    assert "mean_iou" in response.json()["metrics"]


def test_evaluate_reports_loss_under_requested_loss_config(tmp_path: Path) -> None:
    client = _client()
    manifest_path, store_path, checkpoint_path = _train_a_checkpoint(client, tmp_path)

    response = client.post(
        "/evaluate",
        json={
            "manifest_path": str(manifest_path),
            "store": str(store_path),
            "checkpoint_path": str(checkpoint_path),
            "num_classes": 2,
            "loss": {"name": "dice"},
        },
    )

    assert response.status_code == 200
    assert "loss" in response.json()["metrics"]


def test_predict_writes_prediction_files(tmp_path: Path) -> None:
    client = _client()
    manifest_path, store_path, checkpoint_path = _train_a_checkpoint(client, tmp_path)
    output_dir = tmp_path / "predictions"

    response = client.post(
        "/predict",
        json={
            "manifest_path": str(manifest_path),
            "store": str(store_path),
            "checkpoint_path": str(checkpoint_path),
            "num_classes": 2,
            "output_dir": str(output_dir),
        },
    )

    assert response.status_code == 200
    assert len(response.json()["written"]) == 6
    assert len(list(output_dir.glob("*.png"))) == 6


def test_predict_with_overlays_writes_overlay_files(tmp_path: Path) -> None:
    client = _client()
    manifest_path, store_path, checkpoint_path = _train_a_checkpoint(client, tmp_path)
    output_dir = tmp_path / "predictions"

    response = client.post(
        "/predict",
        json={
            "manifest_path": str(manifest_path),
            "store": str(store_path),
            "checkpoint_path": str(checkpoint_path),
            "num_classes": 2,
            "output_dir": str(output_dir),
            "overlays": True,
        },
    )

    assert response.status_code == 200
    assert len(response.json()["written"]) == 12
    assert len(list(output_dir.glob("*_overlay.png"))) == 6


def test_evaluate_missing_checkpoint_returns_404(tmp_path: Path) -> None:
    client = _client()
    manifest_path = build_manifest(tmp_path)
    store_path = tmp_path / "features"
    client.post("/features/build", json={"manifest_path": str(manifest_path), "store": str(store_path)})

    response = client.post(
        "/evaluate",
        json={
            "manifest_path": str(manifest_path),
            "store": str(store_path),
            "checkpoint_path": str(tmp_path / "does-not-exist.ckpt"),
            "num_classes": 2,
        },
    )

    assert response.status_code == 404


def test_predict_empty_sample_set_returns_400(tmp_path: Path) -> None:
    client = _client()
    manifest_path, store_path, checkpoint_path = _train_a_checkpoint(client, tmp_path)

    response = client.post(
        "/predict",
        json={
            "manifest_path": str(manifest_path),
            "store": str(store_path),
            "checkpoint_path": str(checkpoint_path),
            "num_classes": 2,
            "output_dir": str(tmp_path / "predictions"),
            "sample_ids": ["does-not-exist"],
        },
    )

    assert response.status_code == 400


def test_train_rejects_a_config_path_outside_the_api_root(tmp_path_factory: pytest.TempPathFactory) -> None:
    outside = tmp_path_factory.mktemp("outside")
    config_path = outside / "train.yaml"
    config_path.write_text(yaml.safe_dump({"manifest_path": "x", "num_classes": 2}))

    response = _client().post("/train", json={"config_path": str(config_path)})

    assert response.status_code == 400
    assert "config_path" in response.json()["detail"]


def test_predict_rejects_an_output_dir_outside_the_api_root(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    client = _client()
    manifest_path, store_path, checkpoint_path = _train_a_checkpoint(client, tmp_path)
    outside_output_dir = tmp_path_factory.mktemp("outside") / "predictions"

    response = client.post(
        "/predict",
        json={
            "manifest_path": str(manifest_path),
            "store": str(store_path),
            "checkpoint_path": str(checkpoint_path),
            "num_classes": 2,
            "output_dir": str(outside_output_dir),
        },
    )

    assert response.status_code == 400
    assert "output_dir" in response.json()["detail"]


def test_evaluate_rejects_a_checkpoint_path_that_escapes_the_api_root_with_dotdot(tmp_path: Path) -> None:
    client = _client()
    manifest_path, store_path, _ = _train_a_checkpoint(client, tmp_path)

    response = client.post(
        "/evaluate",
        json={
            "manifest_path": str(manifest_path),
            "store": str(store_path),
            "checkpoint_path": "../../escaped.ckpt",
            "num_classes": 2,
        },
    )

    assert response.status_code == 400
    assert "checkpoint_path" in response.json()["detail"]


def test_encoder_benchmark_reports_recommended_settings() -> None:
    client = _client()
    response = client.post("/encoder/benchmark", json={"batch_sizes": [1, 2], "concurrencies": [1], "num_requests": 2})

    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 2
    assert body["best"] is not None
