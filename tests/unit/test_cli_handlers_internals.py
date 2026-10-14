# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.cli.handlers internals not covered by the full CLI
end-to-end tests: the remote-encoder selection branch, the features-build failure path, the
features-clean removal-listing branch, and the encoder-benchmark all-combinations-failed path."""

import argparse
from pathlib import Path

import pytest
import torch

from precisionai.agritune.cli import handlers
from precisionai.agritune.encoder.fake import FakeEncoderBackend, FakeEncoderConfig
from precisionai.agritune.encoder.gateway import EncoderGateway
from precisionai.agritune.encoder.remote import RemoteEncoderBackend
from precisionai.agritune.features.precompute import PrecomputeStats
from precisionai.agritune.features.store import DirectoryFeatureStore
from precisionai.agritune.schemas.features import EncoderFeatures
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
        store_type="directory",
        entries_per_shard=1000,
        augmentation_config=None,
        seed=0,
        base_url=None,
        api_key=None,
        model="pai-embedding",
        preprocessing="",
    )
    exit_code = handlers.features_build(args)

    assert exit_code == 1
    assert "failed sample_ids" in capsys.readouterr().err


def test_features_clean_lists_removed_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store = DirectoryFeatureStore(tmp_path)
    store.write(
        "key1",
        EncoderFeatures(
            patch_tokens=torch.randn(1, 4, 8),
            cls_tokens=None,
            patch_grid=torch.tensor([[2, 2]]),
            valid_patch_mask=None,
            image_sizes=[(224, 224)],
            encoder_model="fake",
            encoder_revision=None,
        ),
    )
    (tmp_path / "key1.json").unlink()  # simulate an interrupted write -> orphaned tensor file

    exit_code = handlers.features_clean(
        argparse.Namespace(store=str(tmp_path), store_type="directory", entries_per_shard=1000)
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "removed 1 file(s)" in out
    assert "key1.safetensors" in out


def test_encoder_benchmark_reports_failure_when_no_combination_succeeds(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    args = argparse.Namespace(
        base_url=None,
        api_key=None,
        model="pai-embedding",
        preprocessing="",
        batch_sizes=[1],
        concurrencies=[1],
        num_requests=2,
        image_size=8,
    )
    # A guaranteed-failing fake backend, injected via monkeypatched _build_raw_encoder to avoid
    # needing a real broken remote endpoint just to exercise this branch.
    failing_backend = FakeEncoderBackend(FakeEncoderConfig(failure_probability=1.0, status_code_on_failure=500))
    original_build_raw_encoder = handlers._build_raw_encoder
    monkeypatch.setattr(
        handlers, "_build_raw_encoder", lambda _args: (failing_backend, original_build_raw_encoder(_args)[1])
    )

    exit_code = handlers.encoder_benchmark(args)

    assert exit_code == 1
    assert "No error-free combination found" in capsys.readouterr().err
