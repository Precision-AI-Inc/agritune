# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.encoder.fake.FakeEncoderBackend."""

import time

import pytest
import torch

from precisionai.agritune.encoder import (
    EncoderRateLimitError,
    EncoderServerError,
    EncoderTimeoutError,
    FakeEncoderBackend,
    FakeEncoderConfig,
    MalformedEncoderResponseError,
)


async def test_encode_returns_features_for_each_image() -> None:
    backend = FakeEncoderBackend(FakeEncoderConfig(patch_dim=16, cls_dim=8, patch_grid=(3, 4)))
    features = await backend.encode(["a", "b", "c"])
    assert features.batch_size == 3
    assert features.patch_tokens.shape == (3, 12, 16)
    assert features.cls_tokens is not None
    assert features.cls_tokens.shape == (3, 8)


async def test_encode_supports_missing_cls_token() -> None:
    backend = FakeEncoderBackend(FakeEncoderConfig(cls_dim=None))
    features = await backend.encode(["a"])
    assert features.cls_tokens is None


async def test_encode_supports_rectangular_grid() -> None:
    backend = FakeEncoderBackend(FakeEncoderConfig(patch_grid=(5, 7)))
    features = await backend.encode(["a"])
    assert features.patch_grid_hw(0) == (5, 7)


async def test_encode_reports_configured_model_and_revision() -> None:
    backend = FakeEncoderBackend(FakeEncoderConfig(encoder_model="my-model", encoder_revision="v2"))
    features = await backend.encode(["a"])
    assert features.encoder_model == "my-model"
    assert features.encoder_revision == "v2"


async def test_encode_same_seed_same_config_is_deterministic_on_first_call() -> None:
    backend_a = FakeEncoderBackend(FakeEncoderConfig(seed=7))
    backend_b = FakeEncoderBackend(FakeEncoderConfig(seed=7))
    features_a = await backend_a.encode(["a"])
    features_b = await backend_b.encode(["a"])
    assert torch.allclose(features_a.patch_tokens, features_b.patch_tokens)


async def test_call_count_increments() -> None:
    backend = FakeEncoderBackend()
    await backend.encode(["a"])
    await backend.encode(["a", "b"])
    assert backend.call_count == 2


async def test_latency_is_applied() -> None:
    # A generous lower bound avoids flakiness from OS timer-resolution jitter around the
    # configured latency, while still catching a regression that drops the sleep entirely.
    backend = FakeEncoderBackend(FakeEncoderConfig(latency_seconds=0.1))
    start = time.monotonic()
    await backend.encode(["a"])
    assert time.monotonic() - start >= 0.05


async def test_failure_probability_429_raises_rate_limit_error() -> None:
    backend = FakeEncoderBackend(FakeEncoderConfig(failure_probability=1.0, status_code_on_failure=429))
    with pytest.raises(EncoderRateLimitError, match="simulated 429"):
        await backend.encode(["a"])


async def test_failure_probability_500_raises_server_error() -> None:
    backend = FakeEncoderBackend(FakeEncoderConfig(failure_probability=1.0, status_code_on_failure=500))
    with pytest.raises(EncoderServerError, match="simulated 500"):
        await backend.encode(["a"])


async def test_rate_limit_error_carries_retry_after() -> None:
    backend = FakeEncoderBackend(FakeEncoderConfig(failure_probability=1.0, status_code_on_failure=429))
    with pytest.raises(EncoderRateLimitError) as exc_info:
        await backend.encode(["a"])
    assert exc_info.value.retry_after_seconds == 1.0
    assert exc_info.value.status_code == 429


async def test_timeout_probability_raises_timeout_error() -> None:
    backend = FakeEncoderBackend(FakeEncoderConfig(timeout_probability=1.0))
    with pytest.raises(EncoderTimeoutError, match="simulated timeout"):
        await backend.encode(["a"])


async def test_malformed_response_probability_raises() -> None:
    backend = FakeEncoderBackend(FakeEncoderConfig(malformed_response_probability=1.0))
    with pytest.raises(MalformedEncoderResponseError, match="malformed response"):
        await backend.encode(["a"])


async def test_no_failure_by_default() -> None:
    backend = FakeEncoderBackend()
    features = await backend.encode(["a", "b"])
    assert features.batch_size == 2
