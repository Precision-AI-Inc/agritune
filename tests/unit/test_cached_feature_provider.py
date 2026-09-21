# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.features.provider.CachedFeatureProvider.

Definition of done this covers: training can run entirely disconnected from the network using a
completed feature store — precompute writes under exactly the keys this provider reads under.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from precisionai.agritune.encoder.fake import FakeEncoderBackend
from precisionai.agritune.features.errors import FeatureNotCachedError
from precisionai.agritune.features.keys import EncoderFingerprint
from precisionai.agritune.features.precompute import precompute_features
from precisionai.agritune.features.provider import CachedFeatureProvider
from precisionai.agritune.features.store import DirectoryFeatureStore
from precisionai.agritune.schemas.protocols import FeatureProvider
from precisionai.agritune.schemas.samples import PreparedSample

_FINGERPRINT = EncoderFingerprint(model="fake-encoder", revision="fake-v1", preprocessing="resize=224")


def _hash_fn(sample: PreparedSample) -> str:
    return sample.image


def _samples(n: int) -> list[PreparedSample]:
    return [PreparedSample(sample_id=f"s{i}", image=f"image-{i}", target=f"mask-{i}") for i in range(n)]


async def _build_store(tmp_path: Path, samples: list[PreparedSample]) -> DirectoryFeatureStore:
    store = DirectoryFeatureStore(tmp_path)
    await precompute_features(
        samples, encoder=FakeEncoderBackend(), store=store, encoder_fingerprint=_FINGERPRINT, image_hash_fn=_hash_fn
    )
    return store


async def test_get_features_returns_precomputed_batch(tmp_path: Path) -> None:
    samples = _samples(3)
    store = await _build_store(tmp_path, samples)
    provider = CachedFeatureProvider(store, encoder_fingerprint=_FINGERPRINT, image_hash_fn=_hash_fn)

    features = provider.get_features(samples)

    assert features.batch_size == 3
    assert features.encoder_model == "fake-encoder"


async def test_get_features_empty_raises(tmp_path: Path) -> None:
    store = DirectoryFeatureStore(tmp_path)
    provider = CachedFeatureProvider(store, encoder_fingerprint=_FINGERPRINT, image_hash_fn=_hash_fn)
    with pytest.raises(ValueError, match="samples must be non-empty"):
        provider.get_features([])


async def test_get_features_missing_sample_raises_feature_not_cached_error(tmp_path: Path) -> None:
    samples = _samples(2)
    store = await _build_store(tmp_path, samples[:1])  # only s0 is cached
    provider = CachedFeatureProvider(store, encoder_fingerprint=_FINGERPRINT, image_hash_fn=_hash_fn)

    with pytest.raises(FeatureNotCachedError, match="s1"):
        provider.get_features(samples)


def test_cached_feature_provider_satisfies_protocol(tmp_path: Path) -> None:
    store = DirectoryFeatureStore(tmp_path)
    provider = CachedFeatureProvider(store, encoder_fingerprint=_FINGERPRINT, image_hash_fn=_hash_fn)
    assert isinstance(provider, FeatureProvider)


async def test_key_for_matches_the_key_precompute_wrote_under(tmp_path: Path) -> None:
    samples = _samples(1)
    store = await _build_store(tmp_path, samples)
    provider = CachedFeatureProvider(store, encoder_fingerprint=_FINGERPRINT, image_hash_fn=_hash_fn)

    assert store.has(provider.key_for(samples[0]))


async def test_different_encoder_fingerprint_misses_cache(tmp_path: Path) -> None:
    samples = _samples(1)
    store = await _build_store(tmp_path, samples)
    other_fingerprint = EncoderFingerprint(model="different-model", revision=None, preprocessing="resize=224")
    provider = CachedFeatureProvider(store, encoder_fingerprint=other_fingerprint, image_hash_fn=_hash_fn)

    with pytest.raises(FeatureNotCachedError):
        provider.get_features(samples)


async def test_max_read_workers_bounds_the_read_thread_pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    samples = _samples(3)
    store = await _build_store(tmp_path, samples)
    provider = CachedFeatureProvider(
        store, encoder_fingerprint=_FINGERPRINT, image_hash_fn=_hash_fn, max_read_workers=2
    )

    seen_max_workers: list[int | None] = []
    original_init = ThreadPoolExecutor.__init__

    def spy_init(self: ThreadPoolExecutor, *args: object, max_workers: int | None = None, **kwargs: object) -> None:
        seen_max_workers.append(max_workers)
        original_init(self, *args, max_workers=max_workers, **kwargs)

    monkeypatch.setattr(ThreadPoolExecutor, "__init__", spy_init)

    provider.get_features(samples)

    # min(max_read_workers, len(samples)) = min(2, 3) = 2.
    assert seen_max_workers == [2]


async def test_default_max_read_workers_does_not_restrict_a_small_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = _samples(3)
    store = await _build_store(tmp_path, samples)
    provider = CachedFeatureProvider(store, encoder_fingerprint=_FINGERPRINT, image_hash_fn=_hash_fn)

    seen_max_workers: list[int | None] = []
    original_init = ThreadPoolExecutor.__init__

    def spy_init(self: ThreadPoolExecutor, *args: object, max_workers: int | None = None, **kwargs: object) -> None:
        seen_max_workers.append(max_workers)
        original_init(self, *args, max_workers=max_workers, **kwargs)

    monkeypatch.setattr(ThreadPoolExecutor, "__init__", spy_init)

    provider.get_features(samples)

    # min(default _MAX_READ_WORKERS=32, len(samples)) = min(32, 3) = 3.
    assert seen_max_workers == [3]
