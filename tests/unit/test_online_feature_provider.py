# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.features.provider.OnlineFeatureProvider."""

import pytest

from precisionai.agritune.encoder.fake import FakeEncoderBackend
from precisionai.agritune.encoder.gateway import EncoderGateway
from precisionai.agritune.features.provider import OnlineFeatureProvider
from precisionai.agritune.schemas.protocols import FeatureProvider
from precisionai.agritune.schemas.samples import PreparedSample


def _samples(n: int) -> list[PreparedSample]:
    return [PreparedSample(sample_id=f"s{i}", image=f"image-{i}", target=None) for i in range(n)]


def test_get_features_encodes_through_the_gateway() -> None:
    provider = OnlineFeatureProvider(EncoderGateway(FakeEncoderBackend()))

    features = provider.get_features(_samples(3))

    assert features.batch_size == 3
    assert features.encoder_model == "fake-encoder"


def test_get_features_empty_raises() -> None:
    provider = OnlineFeatureProvider(EncoderGateway(FakeEncoderBackend()))
    with pytest.raises(ValueError, match="samples must be non-empty"):
        provider.get_features([])


def test_get_features_re_encodes_every_call_with_no_cache() -> None:
    backend = FakeEncoderBackend()
    provider = OnlineFeatureProvider(EncoderGateway(backend))

    provider.get_features(_samples(2))
    provider.get_features(_samples(2))

    assert backend.call_count == 2


def test_online_feature_provider_satisfies_protocol() -> None:
    provider = OnlineFeatureProvider(EncoderGateway(FakeEncoderBackend()))
    assert isinstance(provider, FeatureProvider)
