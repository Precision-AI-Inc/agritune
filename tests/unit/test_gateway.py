# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.encoder.gateway.EncoderGateway.

Phase 5 definition of done: a fake encoder can simulate 429/500/timeout/malformed responses and
the gateway behaves predictably (retries transient failures, gives up after max_attempts, tracks
metrics, and flags a mid-run encoder dimension drift).
"""

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

import pytest
import torch

from precisionai.agritune.encoder.errors import EncoderRateLimitError, EncoderServerError, EncoderTimeoutError
from precisionai.agritune.encoder.fake import FakeEncoderBackend, FakeEncoderConfig
from precisionai.agritune.encoder.gateway import EncoderGateway, GatewayConfig
from precisionai.agritune.encoder.retry import RetryExhaustedError, RetryPolicyConfig
from precisionai.agritune.encoder.validation import EncoderConsistencyError
from precisionai.agritune.schemas.features import EncoderFeatures
from precisionai.agritune.schemas.protocols import EncoderBackend


def _features(batch_size: int, *, patch_dim: int = 4) -> EncoderFeatures:
    return EncoderFeatures(
        patch_tokens=torch.randn(batch_size, 4, patch_dim),
        cls_tokens=None,
        patch_grid=torch.tensor([[2, 2]] * batch_size),
        valid_patch_mask=None,
        image_sizes=[(224, 224)] * batch_size,
        encoder_model="scripted",
        encoder_revision=None,
    )


class _ConcurrencyBackend:
    """Record in-flight calls and complete them out of order."""

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def encode(self, images: Sequence[Any]) -> EncoderFeatures:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        value = int(images[0])
        await asyncio.sleep(0.01 if value == 0 else 0.001)
        self.active -= 1
        features = _features(len(images))
        features.patch_tokens.fill_(value)
        return features


class _ScriptedBackend:
    """Raises/returns a scripted sequence of outcomes, one per call to ``encode``."""

    def __init__(self, script: list[Exception | int]) -> None:
        self._script = list(script)
        self.call_count = 0

    async def encode(self, images: Sequence[Any]) -> EncoderFeatures:
        self.call_count += 1
        outcome = self._script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _features(len(images), patch_dim=outcome)


def _no_backoff_config(**overrides: Any) -> GatewayConfig:
    retry = RetryPolicyConfig(max_attempts=3, initial_backoff_seconds=0.0, max_backoff_seconds=0.0, jitter_ratio=0.0)
    return GatewayConfig(retry=retry, **overrides)


async def test_gateway_satisfies_encoder_backend_protocol() -> None:
    gateway = EncoderGateway(FakeEncoderBackend())
    assert isinstance(gateway, EncoderBackend)


async def test_empty_images_raises() -> None:
    gateway = EncoderGateway(FakeEncoderBackend())
    with pytest.raises(ValueError, match="images must be non-empty"):
        await gateway.encode([])


async def test_splits_large_batch_and_tracks_metrics() -> None:
    backend = FakeEncoderBackend(FakeEncoderConfig(patch_grid=(2, 2)))
    gateway = EncoderGateway(backend, GatewayConfig(max_batch_images=4))

    features = await gateway.encode(list(range(10)))

    assert features.batch_size == 10
    assert backend.call_count == 3  # ceil(10 / 4)
    assert gateway.metrics.requests_total == 3
    assert gateway.metrics.requests_failed == 0
    assert gateway.metrics.images_encoded == 10


async def test_split_batches_run_concurrently_but_preserve_input_order() -> None:
    backend = _ConcurrencyBackend()
    gateway = EncoderGateway(backend, GatewayConfig(max_batch_images=1, max_concurrency=2))

    features = await gateway.encode([0, 1, 2])

    assert backend.max_active == 2
    assert features.patch_tokens[:, 0, 0].tolist() == [0.0, 1.0, 2.0]


async def test_retries_transient_failure_then_succeeds() -> None:
    backend = _ScriptedBackend([EncoderServerError("boom", status_code=500), 4])
    gateway = EncoderGateway(backend, _no_backoff_config())

    features = await gateway.encode([1, 2])

    assert features.batch_size == 2
    assert backend.call_count == 2
    assert gateway.metrics.requests_total == 2
    assert gateway.metrics.requests_retried == 1
    assert gateway.metrics.requests_failed == 0


async def test_rate_limited_failure_is_counted() -> None:
    backend = _ScriptedBackend([EncoderRateLimitError("limited", status_code=429), 4])
    gateway = EncoderGateway(backend, _no_backoff_config())

    await gateway.encode([1])

    assert gateway.metrics.rate_limited_count == 1
    assert gateway.metrics.requests_retried == 1


async def test_exhausts_retries_and_records_failure() -> None:
    backend = _ScriptedBackend(
        [
            EncoderTimeoutError("slow"),
            EncoderTimeoutError("slow"),
            EncoderTimeoutError("slow"),
        ]
    )
    gateway = EncoderGateway(backend, _no_backoff_config())

    with pytest.raises(RetryExhaustedError):
        await gateway.encode([1])

    assert gateway.metrics.requests_failed == 1
    assert gateway.metrics.timeout_count == 3
    assert gateway.metrics.requests_retried == 2  # first two failures retried, third exhausts


async def test_retry_attempts_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    """Every retry attempt must be visible in logs — a bounded retry-then-fail run and a hang
    look identical from the terminal otherwise."""
    backend = _ScriptedBackend([EncoderTimeoutError("slow"), EncoderTimeoutError("slow"), EncoderTimeoutError("slow")])
    gateway = EncoderGateway(backend, _no_backoff_config())

    with caplog.at_level(logging.WARNING, logger="agritune.encoder.gateway"), pytest.raises(RetryExhaustedError):
        await gateway.encode([1])

    assert caplog.text.count("retrying with backoff") == 2
    assert caplog.text.count("no attempts remaining") == 1


async def test_dimension_drift_across_batches_raises_consistency_error() -> None:
    backend = _ScriptedBackend([4, 8])  # second batch reports a different patch_dim
    gateway = EncoderGateway(backend, GatewayConfig(max_batch_images=1))

    with pytest.raises(EncoderConsistencyError):
        await gateway.encode([1, 2])


async def test_images_per_second_is_zero_before_any_request() -> None:
    gateway = EncoderGateway(FakeEncoderBackend())
    assert gateway.metrics.images_per_second == 0.0


async def test_images_per_second_reflects_throughput() -> None:
    # A tiny simulated latency guarantees a measurable elapsed time on any timer resolution.
    gateway = EncoderGateway(FakeEncoderBackend(FakeEncoderConfig(latency_seconds=0.01)))
    await gateway.encode([1, 2, 3])
    assert gateway.metrics.images_per_second > 0.0
