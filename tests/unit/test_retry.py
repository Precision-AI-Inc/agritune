# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.encoder.retry."""

import random

import pytest

from precisionai.agritune.encoder import retry as retry_module
from precisionai.agritune.encoder.errors import (
    EncoderError,
    EncoderRateLimitError,
    EncoderServerError,
    EncoderTimeoutError,
    MalformedEncoderResponseError,
)
from precisionai.agritune.encoder.retry import RetryExhaustedError, RetryPolicy, RetryPolicyConfig


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace asyncio.sleep with an instant, call-recording stand-in so tests run fast."""
    recorded: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        recorded.append(seconds)

    monkeypatch.setattr(retry_module.asyncio, "sleep", fake_sleep)
    return recorded


async def test_succeeds_on_first_attempt_without_sleeping(no_real_sleep: list[float]) -> None:
    policy = RetryPolicy()

    async def operation() -> str:
        return "ok"

    result = await policy.run(operation)
    assert result == "ok"
    assert no_real_sleep == []


async def test_retries_until_success() -> None:
    policy = RetryPolicy(RetryPolicyConfig(max_attempts=3))
    attempts = {"count": 0}

    async def operation() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise EncoderServerError("boom", status_code=500)
        return "recovered"

    result = await policy.run(operation)
    assert result == "recovered"
    assert attempts["count"] == 3


async def test_exhausts_after_max_attempts() -> None:
    policy = RetryPolicy(RetryPolicyConfig(max_attempts=2))

    async def operation() -> str:
        raise EncoderTimeoutError("timed out")

    with pytest.raises(RetryExhaustedError, match="failed after 2 attempt"):
        await policy.run(operation)


async def test_non_retryable_error_propagates_immediately() -> None:
    policy = RetryPolicy(RetryPolicyConfig(max_attempts=5))
    attempts = {"count": 0}

    async def operation() -> str:
        attempts["count"] += 1
        raise MalformedEncoderResponseError("bad shape")

    with pytest.raises(MalformedEncoderResponseError):
        await policy.run(operation)
    assert attempts["count"] == 1


async def test_on_attempt_start_called_for_every_attempt() -> None:
    policy = RetryPolicy(RetryPolicyConfig(max_attempts=3))
    seen_attempts: list[int] = []
    attempts = {"count": 0}

    async def operation() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise EncoderServerError("boom", status_code=500)
        return "ok"

    await policy.run(operation, on_attempt_start=seen_attempts.append)
    assert seen_attempts == [1, 2, 3]


async def test_on_attempt_failure_reports_will_retry_correctly() -> None:
    policy = RetryPolicy(RetryPolicyConfig(max_attempts=2))
    calls: list[tuple[int, bool]] = []

    async def operation() -> str:
        raise EncoderRateLimitError("limited", status_code=429)

    def on_failure(_error: EncoderError, attempt: int, will_retry: bool) -> None:
        calls.append((attempt, will_retry))

    with pytest.raises(RetryExhaustedError):
        await policy.run(operation, on_attempt_failure=on_failure)
    assert calls == [(1, True), (2, False)]


def test_backoff_uses_retry_after_when_present() -> None:
    policy = RetryPolicy(RetryPolicyConfig(initial_backoff_seconds=1.0))
    assert policy.compute_backoff_seconds(attempt=1, retry_after_seconds=7.5) == 7.5


def test_backoff_is_exponential_and_capped() -> None:
    policy = RetryPolicy(
        RetryPolicyConfig(initial_backoff_seconds=1.0, max_backoff_seconds=3.0, jitter_ratio=0.0),
        rng=random.Random(0),
    )
    assert policy.compute_backoff_seconds(attempt=1, retry_after_seconds=None) == 1.0
    assert policy.compute_backoff_seconds(attempt=2, retry_after_seconds=None) == 2.0
    assert policy.compute_backoff_seconds(attempt=3, retry_after_seconds=None) == 3.0  # capped, would be 4.0
    assert policy.compute_backoff_seconds(attempt=10, retry_after_seconds=None) == 3.0


def test_backoff_jitter_stays_within_expected_bounds() -> None:
    policy = RetryPolicy(
        RetryPolicyConfig(initial_backoff_seconds=2.0, max_backoff_seconds=100.0, jitter_ratio=0.2),
        rng=random.Random(0),
    )
    for _ in range(20):
        backoff = policy.compute_backoff_seconds(attempt=1, retry_after_seconds=None)
        assert 1.6 <= backoff <= 2.4
