# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.encoder.rate_limiter."""

import asyncio
import time

import pytest

from precisionai.agritune.encoder.rate_limiter import RateLimiter, RateLimiterConfig


async def test_concurrency_limit_bounds_simultaneous_holders() -> None:
    limiter = RateLimiter(RateLimiterConfig(max_concurrency=2))
    active = 0
    max_active = 0
    lock = asyncio.Lock()

    async def worker() -> None:
        nonlocal active, max_active
        async with limiter.acquire(num_images=1):
            async with lock:
                active += 1
                max_active = max(max_active, active)
            await asyncio.sleep(0.02)
            async with lock:
                active -= 1

    await asyncio.gather(*(worker() for _ in range(6)))
    assert max_active <= 2


async def test_default_config_has_no_throughput_caps() -> None:
    limiter = RateLimiter()
    start = time.monotonic()
    async with limiter.acquire(num_images=1000):
        pass
    assert time.monotonic() - start < 0.5


async def test_requests_per_minute_throttles() -> None:
    # 120 requests/minute = 1 every 0.5s; the second acquire must wait roughly that long.
    limiter = RateLimiter(RateLimiterConfig(max_concurrency=10, requests_per_minute=120))
    async with limiter.acquire(num_images=1):
        pass
    start = time.monotonic()
    async with limiter.acquire(num_images=1):
        pass
    assert time.monotonic() - start >= 0.2


async def test_images_per_minute_throttles() -> None:
    limiter = RateLimiter(RateLimiterConfig(max_concurrency=10, images_per_minute=60))  # 1/sec
    async with limiter.acquire(num_images=1):
        pass
    start = time.monotonic()
    async with limiter.acquire(num_images=5):
        pass
    assert time.monotonic() - start >= 2.0


def test_invalid_max_concurrency_raises() -> None:
    with pytest.raises(ValueError, match="max_concurrency must be positive"):
        RateLimiter(RateLimiterConfig(max_concurrency=0))


def test_non_positive_requests_per_minute_raises() -> None:
    with pytest.raises(ValueError, match="rate_per_second must be positive"):
        RateLimiter(RateLimiterConfig(requests_per_minute=0))
