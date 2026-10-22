# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.features.precompute.

Phase 6 definition of done: training can run entirely disconnected from the network using a
completed feature store, and an interrupted/restarted build only encodes what's still missing.
"""

from collections.abc import Sequence
from pathlib import Path

import pytest
import torch

from precisionai.agritune.encoder.errors import EncoderServerError
from precisionai.agritune.encoder.fake import FakeEncoderBackend, FakeEncoderConfig
from precisionai.agritune.features.keys import EncoderFingerprint, compute_feature_key, hash_augmentation
from precisionai.agritune.features.precompute import PrecomputeStats, precompute_features
from precisionai.agritune.features.store import DirectoryFeatureStore, ShardedFeatureStore
from precisionai.agritune.schemas.features import EncoderFeatures, select_one
from precisionai.agritune.schemas.samples import PreparedSample

_FINGERPRINT = EncoderFingerprint(model="fake-encoder", revision="fake-v1", preprocessing="resize=224")


def _samples(n: int) -> list[PreparedSample]:
    return [PreparedSample(sample_id=f"s{i}", image=f"image-bytes-{i}", target=f"mask-{i}") for i in range(n)]


def _hash_fn(sample: PreparedSample) -> str:
    return sample.image  # already a stand-in string in these fixtures


async def test_all_new_samples_are_computed(tmp_path: Path) -> None:
    store = DirectoryFeatureStore(tmp_path)
    stats = await precompute_features(
        _samples(3),
        encoder=FakeEncoderBackend(),
        store=store,
        encoder_fingerprint=_FINGERPRINT,
        image_hash_fn=_hash_fn,
    )
    assert stats == PrecomputeStats(total=3, computed=3, skipped=0, failed=0, failed_sample_ids=[])
    assert len(store.list_keys()) == 3


async def test_rerun_skips_already_computed_samples(tmp_path: Path) -> None:
    store = DirectoryFeatureStore(tmp_path)
    samples = _samples(3)
    await precompute_features(
        samples, encoder=FakeEncoderBackend(), store=store, encoder_fingerprint=_FINGERPRINT, image_hash_fn=_hash_fn
    )

    second_run_backend = FakeEncoderBackend()
    stats = await precompute_features(
        samples,
        encoder=second_run_backend,
        store=store,
        encoder_fingerprint=_FINGERPRINT,
        image_hash_fn=_hash_fn,
    )

    assert stats.skipped == 3
    assert stats.computed == 0
    assert second_run_backend.call_count == 0


async def test_partial_completion_only_encodes_the_missing_samples(tmp_path: Path) -> None:
    store = DirectoryFeatureStore(tmp_path)
    all_samples = _samples(5)
    await precompute_features(
        all_samples[:3],
        encoder=FakeEncoderBackend(),
        store=store,
        encoder_fingerprint=_FINGERPRINT,
        image_hash_fn=_hash_fn,
    )

    resume_backend = FakeEncoderBackend()
    stats = await precompute_features(
        all_samples, encoder=resume_backend, store=store, encoder_fingerprint=_FINGERPRINT, image_hash_fn=_hash_fn
    )

    assert stats.skipped == 3
    assert stats.computed == 2
    # The two missing samples are encoded together, in a single call.
    assert resume_backend.call_count == 1


async def test_encoder_failure_is_recorded_and_does_not_abort_the_run(tmp_path: Path) -> None:
    store = DirectoryFeatureStore(tmp_path)
    backend = FakeEncoderBackend(FakeEncoderConfig(failure_probability=1.0, status_code_on_failure=500))
    stats = await precompute_features(
        _samples(2), encoder=backend, store=store, encoder_fingerprint=_FINGERPRINT, image_hash_fn=_hash_fn
    )
    assert stats.failed == 2
    assert stats.failed_sample_ids == ["s0", "s1"]
    assert stats.computed == 0


async def test_on_progress_called_once_per_sample(tmp_path: Path) -> None:
    store = DirectoryFeatureStore(tmp_path)
    calls: list[int] = []
    await precompute_features(
        _samples(4),
        encoder=FakeEncoderBackend(),
        store=store,
        encoder_fingerprint=_FINGERPRINT,
        image_hash_fn=_hash_fn,
        on_progress=lambda stats: calls.append(stats.computed + stats.skipped + stats.failed),
    )
    assert calls == [1, 2, 3, 4]


async def test_final_flush_persists_trailing_partial_shard(tmp_path: Path) -> None:
    store = ShardedFeatureStore(tmp_path, entries_per_shard=10)
    await precompute_features(
        _samples(3), encoder=FakeEncoderBackend(), store=store, encoder_fingerprint=_FINGERPRINT, image_hash_fn=_hash_fn
    )
    assert (tmp_path / "shard_00000.safetensors").exists()
    assert len(store.list_keys()) == 3


class _RecordingEncoder:
    """Fake encoder that records each call's batch size and rejects one image."""

    def __init__(self, reject: str | None = None) -> None:
        self.inner = FakeEncoderBackend()
        self.calls: list[int] = []
        self.reject = reject

    async def encode(self, images: Sequence[str]) -> EncoderFeatures:
        self.calls.append(len(images))
        if self.reject in images:
            raise EncoderServerError("rejected", status_code=500)
        return await self.inner.encode(images)


def _key(sample: PreparedSample) -> str:
    return compute_feature_key(
        sample_id=sample.sample_id,
        image_hash=_hash_fn(sample),
        augmentation_fingerprint=hash_augmentation(sample.augmentation_metadata),
        encoder_fingerprint=_FINGERPRINT,
    )


async def test_uncached_samples_are_encoded_in_groups(tmp_path: Path) -> None:
    store = DirectoryFeatureStore(tmp_path)
    encoder = _RecordingEncoder()
    samples = _samples(10)

    stats = await precompute_features(
        samples, encoder=encoder, store=store, encoder_fingerprint=_FINGERPRINT, image_hash_fn=_hash_fn, group_size=4
    )

    assert stats.computed == 10
    assert encoder.calls == [4, 4, 2]
    # The fake encoder seeds by call, so a fresh one reproduces the first group's batch; each sample gets its own slice.
    first_group = await FakeEncoderBackend().encode([sample.image for sample in samples[:4]])
    for index, sample in enumerate(samples[:4]):
        assert torch.equal(store.read(_key(sample)).patch_tokens, select_one(first_group, index).patch_tokens)


async def test_a_rejected_image_fails_alone_while_its_group_is_written(tmp_path: Path) -> None:
    store = DirectoryFeatureStore(tmp_path)
    encoder = _RecordingEncoder(reject="image-bytes-2")

    stats = await precompute_features(
        _samples(4),
        encoder=encoder,
        store=store,
        encoder_fingerprint=_FINGERPRINT,
        image_hash_fn=_hash_fn,
        group_size=4,
    )

    assert stats.computed == 3
    assert stats.failed_sample_ids == ["s2"]
    assert encoder.calls == [4, 1, 1, 1, 1]


async def test_group_size_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="group_size must be positive"):
        await precompute_features(
            _samples(1),
            encoder=FakeEncoderBackend(),
            store=DirectoryFeatureStore(tmp_path),
            encoder_fingerprint=_FINGERPRINT,
            image_hash_fn=_hash_fn,
            group_size=0,
        )
