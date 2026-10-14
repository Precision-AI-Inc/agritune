# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the dataset routes, via ``fastapi.testclient.TestClient``."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from precisionai.agritune.api import create_app
from tests.fixtures.manifest_factory import build_manifest


@pytest.fixture(autouse=True)
def _api_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Scope the API's path-containment root to this test's ``tmp_path``."""
    monkeypatch.setenv("AGRITUNE_API_ROOT", str(tmp_path))


def _client() -> TestClient:
    return TestClient(create_app())


def test_validate_reports_ok_for_a_clean_manifest(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path)
    response = _client().post("/dataset/validate", json={"manifest_path": str(manifest_path)})
    assert response.status_code == 200
    assert response.json() == {"is_valid": True, "issues": []}


def test_validate_reports_issues_for_a_broken_manifest(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path, omit_image_for="sample-1")
    response = _client().post("/dataset/validate", json={"manifest_path": str(manifest_path)})
    assert response.status_code == 200
    body = response.json()
    assert body["is_valid"] is False
    assert any(issue["category"] == "missing_image" for issue in body["issues"])


def test_validate_missing_manifest_returns_404(tmp_path: Path) -> None:
    response = _client().post("/dataset/validate", json={"manifest_path": str(tmp_path / "nope.csv")})
    assert response.status_code == 404
    assert "detail" in response.json()


def test_inspect_reports_dataset_statistics(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path)
    response = _client().get("/dataset/inspect", params={"manifest_path": str(manifest_path)})
    assert response.status_code == 200
    body = response.json()
    assert body["num_samples"] == 4
    assert body["metadata_columns"] == ["field_id"]


def test_validate_rejects_a_manifest_path_outside_the_api_root(tmp_path_factory: pytest.TempPathFactory) -> None:
    outside = tmp_path_factory.mktemp("outside")
    manifest_path = build_manifest(outside)
    response = _client().post("/dataset/validate", json={"manifest_path": str(manifest_path)})
    assert response.status_code == 400
    assert "manifest_path" in response.json()["detail"]


def test_validate_rejects_a_relative_path_that_escapes_the_api_root() -> None:
    response = _client().post("/dataset/validate", json={"manifest_path": "../escaped.csv"})
    assert response.status_code == 400
    assert "manifest_path" in response.json()["detail"]
