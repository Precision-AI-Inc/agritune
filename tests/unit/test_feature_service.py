# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.services.feature_service."""

from pathlib import Path

from precisionai.agritune.encoder.fake import FakeEncoderBackend
from precisionai.agritune.features.keys import EncoderFingerprint
from precisionai.agritune.features.store import DirectoryFeatureStore
from precisionai.agritune.services.feature_service import build_features
from tests.fixtures.manifest_factory import build_manifest

_FINGERPRINT = EncoderFingerprint(model="fake-encoder", revision="fake-v1", preprocessing="resize=8x8")


async def test_build_features_computes_every_sample(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path)
    store = DirectoryFeatureStore(tmp_path / "features")

    stats = await build_features(
        str(manifest_path), store=store, encoder=FakeEncoderBackend(), encoder_fingerprint=_FINGERPRINT
    )

    assert stats.computed == 4
    assert stats.failed == 0
    assert len(store.list_keys()) == 4


async def test_build_features_restricts_to_given_sample_ids(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path)
    store = DirectoryFeatureStore(tmp_path / "features")

    stats = await build_features(
        str(manifest_path),
        store=store,
        encoder=FakeEncoderBackend(),
        encoder_fingerprint=_FINGERPRINT,
        sample_ids=["sample-0", "sample-1"],
    )

    assert stats.total == 2
    assert stats.computed == 2
    assert len(store.list_keys()) == 2


async def test_build_features_is_resumable(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path)
    store = DirectoryFeatureStore(tmp_path / "features")

    await build_features(
        str(manifest_path), store=store, encoder=FakeEncoderBackend(), encoder_fingerprint=_FINGERPRINT
    )

    second_backend = FakeEncoderBackend()
    stats = await build_features(
        str(manifest_path), store=store, encoder=second_backend, encoder_fingerprint=_FINGERPRINT
    )

    assert stats.skipped == 4
    assert stats.computed == 0
    assert second_backend.call_count == 0
