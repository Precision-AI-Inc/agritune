# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.features.provider.HybridFeatureProvider.

``get_features`` bridges the async gateway to the synchronous ``FeatureProvider`` interface via
``asyncio.run`` internally, so every test here calls it from plain (non-``async def``) test
functions — the same way ``Trainer`` calls it in production. Async setup (precomputing a store)
is driven explicitly via ``asyncio.run`` before that point.
"""

import asyncio
from pathlib import Path

import pytest

from precisionai.agritune.encoder.fake import FakeEncoderBackend
from precisionai.agritune.encoder.gateway import EncoderGateway
from precisionai.agritune.features.keys import EncoderFingerprint
from precisionai.agritune.features.precompute import precompute_features
from precisionai.agritune.features.provider import HybridFeatureProvider
from precisionai.agritune.features.store import DirectoryFeatureStore
from precisionai.agritune.schemas.protocols import FeatureProvider
from precisionai.agritune.schemas.samples import PreparedSample

_FINGERPRINT = EncoderFingerprint(model="fake-encoder", revision="fake-v1", preprocessing="resize=224")


def _hash_fn(sample: PreparedSample) -> str:
    return sample.image


def _samples(n: int) -> list[PreparedSample]:
    return [PreparedSample(sample_id=f"s{i}", image=f"image-{i}", target=None) for i in range(n)]


def _precompute(tmp_path: Path, samples: list[PreparedSample]) -> DirectoryFeatureStore:
    store = DirectoryFeatureStore(tmp_path)
    asyncio.run(
        precompute_features(
            samples,
            encoder=FakeEncoderBackend(),
            store=store,
            encoder_fingerprint=_FINGERPRINT,
            image_hash_fn=_hash_fn,
        )
    )
    return store


def test_all_hits_never_calls_the_encoder(tmp_path: Path) -> None:
    samples = _samples(3)
    store = _precompute(tmp_path, samples)
    backend = FakeEncoderBackend()
    provider = HybridFeatureProvider(
        store, gateway=EncoderGateway(backend), encoder_fingerprint=_FINGERPRINT, image_hash_fn=_hash_fn
    )

    features = provider.get_features(samples)

    assert features.batch_size == 3
    assert backend.call_count == 0


def test_all_misses_encodes_and_writes_through(tmp_path: Path) -> None:
    store = DirectoryFeatureStore(tmp_path)
    backend = FakeEncoderBackend()
    provider = HybridFeatureProvider(
        store, gateway=EncoderGateway(backend), encoder_fingerprint=_FINGERPRINT, image_hash_fn=_hash_fn
    )
    samples = _samples(2)

    features = provider.get_features(samples)

    assert features.batch_size == 2
    assert backend.call_count == 1
    for sample in samples:
        assert store.has(provider.key_for(sample))


def test_a_miss_is_never_re_encoded_on_a_later_call(tmp_path: Path) -> None:
    store = DirectoryFeatureStore(tmp_path)
    backend = FakeEncoderBackend()
    provider = HybridFeatureProvider(
        store, gateway=EncoderGateway(backend), encoder_fingerprint=_FINGERPRINT, image_hash_fn=_hash_fn
    )
    samples = _samples(1)

    provider.get_features(samples)  # first call: miss, encodes and writes through
    provider.get_features(samples)  # second call: must now be a cache hit

    assert backend.call_count == 1


def test_mixed_hits_and_misses_preserve_input_order(tmp_path: Path) -> None:
    samples = _samples(3)
    store = _precompute(tmp_path, samples[:1])  # only s0 is precomputed
    backend = FakeEncoderBackend()
    provider = HybridFeatureProvider(
        store, gateway=EncoderGateway(backend), encoder_fingerprint=_FINGERPRINT, image_hash_fn=_hash_fn
    )

    features = provider.get_features(samples)

    assert features.batch_size == 3
    assert backend.call_count == 1  # s1 and s2 encoded together in one gateway call
    assert store.has(provider.key_for(samples[1]))
    assert store.has(provider.key_for(samples[2]))


def test_get_features_empty_raises(tmp_path: Path) -> None:
    store = DirectoryFeatureStore(tmp_path)
    provider = HybridFeatureProvider(
        store, gateway=EncoderGateway(FakeEncoderBackend()), encoder_fingerprint=_FINGERPRINT, image_hash_fn=_hash_fn
    )
    with pytest.raises(ValueError, match="samples must be non-empty"):
        provider.get_features([])


def test_hybrid_feature_provider_satisfies_protocol(tmp_path: Path) -> None:
    store = DirectoryFeatureStore(tmp_path)
    provider = HybridFeatureProvider(
        store, gateway=EncoderGateway(FakeEncoderBackend()), encoder_fingerprint=_FINGERPRINT, image_hash_fn=_hash_fn
    )
    assert isinstance(provider, FeatureProvider)
