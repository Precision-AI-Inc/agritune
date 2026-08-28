# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Retry policy with exponential backoff and jitter for transient encoder failures.

Only :class:`~precisionai.agritune.encoder.errors.EncoderRateLimitError`,
:class:`~precisionai.agritune.encoder.errors.EncoderServerError`, and
:class:`~precisionai.agritune.encoder.errors.EncoderTimeoutError` are retried — a malformed
response or a generic client error (e.g. 400/401) is not transient and is raised immediately.
This is training-specific retry behavior and belongs here, not in ``RemoteEncoderBackend``.
"""

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from precisionai.agritune.encoder.errors import (
    EncoderError,
    EncoderRateLimitError,
    EncoderServerError,
    EncoderTimeoutError,
)

T = TypeVar("T")

_RETRYABLE_EXCEPTIONS = (EncoderRateLimitError, EncoderServerError, EncoderTimeoutError)


@dataclass
class RetryPolicyConfig:
    """Configuration for :class:`RetryPolicy`.

    Attributes
    ----------
    max_attempts : int
        Total attempts before giving up (including the first, non-retry attempt).
    initial_backoff_seconds : float
        Backoff before the second attempt; doubles on each subsequent attempt (capped by
        ``max_backoff_seconds``), unless the failure carries a server-supplied ``Retry-After``.
    max_backoff_seconds : float
        Upper bound on computed backoff, before jitter.
    jitter_ratio : float
        Backoff is jittered by up to ``± backoff * jitter_ratio``.
    """

    max_attempts: int = 5
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 30.0
    jitter_ratio: float = 0.1


class RetryExhaustedError(EncoderError):
    """All retry attempts failed; wraps the final underlying :class:`EncoderError`."""


class RetryPolicy:
    """Retries a retryable encoder operation with exponential backoff and jitter.

    Parameters
    ----------
    config : RetryPolicyConfig | None, optional
        Retry configuration; defaults to :class:`RetryPolicyConfig`.
    rng : random.Random | None, optional
        Source of jitter randomness; a dedicated instance for deterministic tests.
    """

    def __init__(self, config: RetryPolicyConfig | None = None, *, rng: random.Random | None = None) -> None:
        self._config = config or RetryPolicyConfig()
        self._rng = rng or random.Random()

    def compute_backoff_seconds(self, *, attempt: int, retry_after_seconds: float | None) -> float:
        """Compute the delay before the given retry attempt.

        Parameters
        ----------
        attempt : int
            The attempt number that just failed (``1`` for the first attempt).
        retry_after_seconds : float | None
            A server-supplied ``Retry-After`` value, when the failure carried one — authoritative
            over the computed exponential backoff.

        Returns
        -------
        float
            Seconds to wait before the next attempt.
        """
        if retry_after_seconds is not None:
            return max(retry_after_seconds, 0.0)
        base = min(self._config.initial_backoff_seconds * (2 ** (attempt - 1)), self._config.max_backoff_seconds)
        jitter = base * self._config.jitter_ratio
        return max(base + self._rng.uniform(-jitter, jitter), 0.0)

    async def run(
        self,
        operation: Callable[[], Awaitable[T]],
        *,
        on_attempt_start: Callable[[int], None] | None = None,
        on_attempt_failure: Callable[[EncoderError, int, bool], None] | None = None,
    ) -> T:
        """Run ``operation``, retrying on transient encoder failures.

        Parameters
        ----------
        operation : Callable[[], Awaitable[T]]
            A zero-argument callable returning an awaitable; called fresh on every attempt.
        on_attempt_start : Callable[[int], None] | None, optional
            Called with the attempt number immediately before every attempt (success or failure)
            — lets a caller count total attempts made.
        on_attempt_failure : Callable[[EncoderError, int, bool], None] | None, optional
            Called after each failed attempt with ``(error, attempt_number, will_retry)`` —
            lets a caller (e.g. :class:`~precisionai.agritune.encoder.gateway.EncoderGateway`)
            track per-attempt metrics.

        Returns
        -------
        T
            The operation's result on the first successful attempt.

        Raises
        ------
        RetryExhaustedError
            If every attempt up to ``max_attempts`` fails.
        """
        last_error: EncoderError | None = None
        for attempt in range(1, self._config.max_attempts + 1):
            if on_attempt_start is not None:
                on_attempt_start(attempt)
            try:
                return await operation()
            except _RETRYABLE_EXCEPTIONS as exc:
                last_error = exc
                will_retry = attempt < self._config.max_attempts
                if on_attempt_failure is not None:
                    on_attempt_failure(exc, attempt, will_retry)
                if not will_retry:
                    break
                retry_after = getattr(exc, "retry_after_seconds", None)
                backoff = self.compute_backoff_seconds(attempt=attempt, retry_after_seconds=retry_after)
                await asyncio.sleep(backoff)

        raise RetryExhaustedError(
            f"encoder call failed after {self._config.max_attempts} attempt(s): {last_error}"
        ) from last_error
