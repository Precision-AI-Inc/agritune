# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.cli.handlers internals not covered by the full CLI
end-to-end tests: the remote-encoder selection branch and the features-build failure path."""

import argparse
from pathlib import Path

import pytest

from precisionai.agritune.cli import handlers
from precisionai.agritune.encoder.gateway import EncoderGateway
from precisionai.agritune.encoder.remote import RemoteEncoderBackend
from precisionai.agritune.features.precompute import PrecomputeStats
from tests.fixtures.manifest_factory import build_manifest


def _encoder_args(*, base_url: str | None) -> argparse.Namespace:
    return argparse.Namespace(base_url=base_url, api_key="sk-pai-test", model="pai-embedding", preprocessing="")


def test_build_encoder_selects_remote_backend_when_base_url_given() -> None:
    gateway, fingerprint = handlers._build_encoder(_encoder_args(base_url="https://example.test/v1"))
    assert isinstance(gateway, EncoderGateway)
    assert isinstance(gateway._backend, RemoteEncoderBackend)
    assert fingerprint.model == "pai-embedding"
    assert fingerprint.revision is None


def test_build_encoder_selects_fake_backend_by_default() -> None:
    gateway, fingerprint = handlers._build_encoder(_encoder_args(base_url=None))
    assert isinstance(gateway, EncoderGateway)
    assert fingerprint.model == "fake-encoder"


async def _failing_build_features(*_args: object, **_kwargs: object) -> PrecomputeStats:
    return PrecomputeStats(total=1, computed=0, skipped=0, failed=1, failed_sample_ids=["sample-0"])


def test_features_build_reports_failure_and_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = build_manifest(tmp_path)
    monkeypatch.setattr(handlers, "build_features", _failing_build_features)

    args = argparse.Namespace(
        manifest=str(manifest_path),
        store=str(tmp_path / "features"),
        base_url=None,
        api_key=None,
        model="pai-embedding",
        preprocessing="",
    )
    exit_code = handlers.features_build(args)

    assert exit_code == 1
    assert "failed sample_ids" in capsys.readouterr().err
