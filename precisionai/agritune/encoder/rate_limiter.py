# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Independently configurable rate limiting: bounded concurrency, requests/minute, images/minute.

Values are never hardcoded — ``agritune encoder benchmark`` is what makes reasonable values
empirical rather than guesses.
"""

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass


@dataclass
class RateLimiterConfig:
    """Configuration for :class:`RateLimiter`.

    Attributes
    ----------
    max_concurrency : int
        Maximum number of encoder requests in flight at once.
    requests_per_minute : float | None
        Maximum sustained request rate; ``None`` disables this limit.
    images_per_minute : float | None
        Maximum sustained image-throughput rate; ``None`` disables this limit.
    """

    max_concurrency: int = 8
    requests_per_minute: float | None = None
    images_per_minute: float | None = None


class _TokenBucket:
    """A simple async token bucket: tokens refill continuously at a fixed rate.

    Base burst capacity is a flat one second's worth of tokens — enough to smooth normal jitter
    without allowing sustained bursting. A single ``acquire`` for more than that (e.g. one large
    batch under a strict per-minute cap) is still satisfied — never a deadlock — by temporarily
    raising the effective ceiling to the requested amount for that call; the amount still costs
    exactly ``amount / rate`` seconds of waiting if the bucket is not already full enough.
    """

    _BASE_CAPACITY = 1.0

    def __init__(self, *, rate_per_second: float) -> None:
        if rate_per_second <= 0:
            raise ValueError(f"rate_per_second must be positive; got {rate_per_second}")
        self._rate = rate_per_second
        self._tokens = self._BASE_CAPACITY
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, amount: float) -> None:
        """Block until ``amount`` tokens are available, then consume them."""
        effective_capacity = max(self._BASE_CAPACITY, amount)
        async with self._lock:
            while True:
                self._refill(effective_capacity)
                if self._tokens >= amount:
                    self._tokens -= amount
                    return
                wait_seconds = (amount - self._tokens) / self._rate
                await asyncio.sleep(wait_seconds)

    def _refill(self, capacity: float) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now


class RateLimiter:
    """Bounds concurrency and, optionally, sustained requests/minute and images/minute.

    Parameters
    ----------
    config : RateLimiterConfig | None, optional
        Defaults to :class:`RateLimiterConfig` (concurrency-only, no throughput caps).
    """

    def __init__(self, config: RateLimiterConfig | None = None) -> None:
        self._config = config or RateLimiterConfig()
        if self._config.max_concurrency <= 0:
            raise ValueError(f"max_concurrency must be positive; got {self._config.max_concurrency}")
        self._concurrency_semaphore = asyncio.Semaphore(self._config.max_concurrency)
        self._request_bucket = (
            _TokenBucket(rate_per_second=self._config.requests_per_minute / 60.0)
            if self._config.requests_per_minute is not None
            else None
        )
        self._image_bucket = (
            _TokenBucket(rate_per_second=self._config.images_per_minute / 60.0)
            if self._config.images_per_minute is not None
            else None
        )

    @asynccontextmanager
    async def acquire(self, *, num_images: int) -> AsyncIterator[None]:
        """Acquire a concurrency slot and (if configured) request/image rate budget.

        Parameters
        ----------
        num_images : int
            Positive number of images the caller is about to send in this request, consumed from
            the images/minute budget.

        Raises
        ------
        ValueError
            If ``num_images`` is not positive.
        """
        if num_images <= 0:
            raise ValueError(f"num_images must be positive; got {num_images}")
        async with self._concurrency_semaphore:
            if self._request_bucket is not None:
                await self._request_bucket.acquire(1.0)
            if self._image_bucket is not None:
                await self._image_bucket.acquire(float(num_images))
            yield
