# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the features routes, via ``fastapi.testclient.TestClient``."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from precisionai.agritune.api import create_app
from precisionai.agritune.features.store import DirectoryFeatureStore
from tests.fixtures.manifest_factory import build_manifest


@pytest.fixture(autouse=True)
def _api_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Scope the API's path-containment root to this test's ``tmp_path``."""
    monkeypatch.setenv("AGRITUNE_API_ROOT", str(tmp_path))


def _client() -> TestClient:
    return TestClient(create_app())


def _build(client: TestClient, tmp_path: Path) -> tuple[Path, Path]:
    manifest_path = build_manifest(tmp_path)
    store_path = tmp_path / "features"
    response = client.post("/features/build", json={"manifest_path": str(manifest_path), "store": str(store_path)})
    assert response.status_code == 200
    return manifest_path, store_path


def test_build_computes_features_for_every_sample(tmp_path: Path) -> None:
    client = _client()
    manifest_path = build_manifest(tmp_path)
    store_path = tmp_path / "features"

    response = client.post("/features/build", json={"manifest_path": str(manifest_path), "store": str(store_path)})

    assert response.status_code == 200
    assert response.json() == {"total": 4, "computed": 4, "skipped": 0, "failed": 0, "failed_sample_ids": []}
    assert store_path.is_dir()


def test_verify_reports_ok_for_a_clean_store(tmp_path: Path) -> None:
    client = _client()
    _, store_path = _build(client, tmp_path)

    response = client.post("/features/verify", json={"store": str(store_path)})

    assert response.status_code == 200
    assert response.json() == {"is_valid": True, "total": 4, "corrupted_keys": []}


def test_verify_detects_corruption(tmp_path: Path) -> None:
    client = _client()
    _, store_path = _build(client, tmp_path)
    store = DirectoryFeatureStore(store_path)
    corrupted_key = store.list_keys()[0]
    with (store_path / f"{corrupted_key}.safetensors").open("r+b") as handle:
        handle.seek(0)
        handle.write(b"\x00" * 16)

    response = client.post("/features/verify", json={"store": str(store_path)})

    assert response.status_code == 200
    body = response.json()
    assert body["is_valid"] is False
    assert corrupted_key in body["corrupted_keys"]


def test_inspect_reports_store_statistics(tmp_path: Path) -> None:
    client = _client()
    _, store_path = _build(client, tmp_path)

    response = client.get("/features/inspect", params={"store": str(store_path)})

    assert response.status_code == 200
    body = response.json()
    assert body["total_entries"] == 4
    assert body["encoder_models"] == {"fake-encoder": 4}


def test_clean_reports_no_removals_for_a_clean_store(tmp_path: Path) -> None:
    client = _client()
    _, store_path = _build(client, tmp_path)

    response = client.post("/features/clean", json={"store": str(store_path)})

    assert response.status_code == 200
    assert response.json() == {"removed": []}


def test_clean_lists_removed_files(tmp_path: Path) -> None:
    client = _client()
    _, store_path = _build(client, tmp_path)
    store = DirectoryFeatureStore(store_path)
    key = store.list_keys()[0]
    (store_path / f"{key}.json").unlink()  # simulate an interrupted write -> orphaned tensor file

    response = client.post("/features/clean", json={"store": str(store_path)})

    assert response.status_code == 200
    assert response.json()["removed"] == [f"{key}.safetensors"]


def test_build_rejects_a_store_path_outside_the_api_root(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    manifest_path = build_manifest(tmp_path)
    outside_store = tmp_path_factory.mktemp("outside") / "features"

    response = _client().post(
        "/features/build", json={"manifest_path": str(manifest_path), "store": str(outside_store)}
    )

    assert response.status_code == 400
    assert "store" in response.json()["detail"]


def test_build_rejects_a_manifest_path_that_escapes_the_api_root_with_dotdot(tmp_path: Path) -> None:
    response = _client().post(
        "/features/build", json={"manifest_path": "../escaped.csv", "store": str(tmp_path / "features")}
    )

    assert response.status_code == 400
    assert "manifest_path" in response.json()["detail"]
