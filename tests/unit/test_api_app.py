# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for precisionai.agritune.api.app.create_app.

Route wiring itself is exercised end-to-end in ``test_api_dataset_routes.py``,
``test_api_features_routes.py``, and ``test_api_run_routes.py``. This file isolates
``create_app()``'s own contribution: every route is registered, and each exception handler maps
its domain exception to the right HTTP status — tested here via throwaway routes that raise the
exception directly, since not every mapped exception is reachable through today's wired services
(e.g. ``CheckpointMismatchError`` is only ever raised by ``Trainer._resume``, which
``training_service.run_training`` does not yet call with fingerprints to compare against).
"""

from fastapi.testclient import TestClient

from precisionai.agritune.api import create_app
from precisionai.agritune.features.errors import FeatureNotCachedError
from precisionai.agritune.training.checkpointing import CheckpointMismatchError


def test_every_route_is_registered() -> None:
    # app.routes wraps included routers opaquely on this FastAPI version, so assert against the
    # generated OpenAPI schema instead — the one place the full route tree is reliably flattened.
    app = create_app()
    paths = set(app.openapi()["paths"])
    assert paths == {
        "/dataset/validate",
        "/dataset/inspect",
        "/features/build",
        "/features/verify",
        "/features/inspect",
        "/features/clean",
        "/train",
        "/evaluate",
        "/predict",
        "/encoder/benchmark",
    }


def _client_raising(exc: Exception) -> TestClient:
    app = create_app()

    @app.get("/_raises")
    def _raises() -> None:
        raise exc

    return TestClient(app, raise_server_exceptions=False)


def test_value_error_maps_to_400() -> None:
    client = _client_raising(ValueError("bad input"))
    response = client.get("/_raises")
    assert response.status_code == 400
    assert response.json() == {"detail": "bad input"}


def test_file_not_found_error_maps_to_404() -> None:
    client = _client_raising(FileNotFoundError("no such file: x.csv"))
    response = client.get("/_raises")
    assert response.status_code == 404
    assert response.json() == {"detail": "no such file: x.csv"}


def test_feature_not_cached_error_maps_to_404() -> None:
    # FeatureNotCachedError subclasses KeyError, whose str() wraps the message in repr() — assert
    # against that same str() rather than hand-computing the quoting.
    exc = FeatureNotCachedError("no cached features for sample s0")
    client = _client_raising(exc)
    response = client.get("/_raises")
    assert response.status_code == 404
    assert response.json() == {"detail": str(exc)}


def test_checkpoint_mismatch_error_maps_to_409() -> None:
    client = _client_raising(CheckpointMismatchError("fingerprint mismatch"))
    response = client.get("/_raises")
    assert response.status_code == 409
    assert response.json() == {"detail": "fingerprint mismatch"}
