# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""``EncoderGateway`` — the infrastructure layer between ``FeatureProvider`` and ``EncoderBackend``.

Conceptually::

    samples -> request queue -> batcher -> rate limiter -> remote encoder -> validation -> feature queue

The gateway itself satisfies :class:`~precisionai.agritune.schemas.protocols.EncoderBackend`, so a
``FeatureProvider`` can hold either a bare backend or a gateway-wrapped one without caring which.
"""

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from precisionai.agritune.encoder.base import EncoderBackend, ImageInput
from precisionai.agritune.encoder.batching import batch_items
from precisionai.agritune.encoder.errors import EncoderError, EncoderRateLimitError, EncoderTimeoutError
from precisionai.agritune.encoder.rate_limiter import RateLimiter, RateLimiterConfig
from precisionai.agritune.encoder.retry import RetryPolicy, RetryPolicyConfig
from precisionai.agritune.encoder.validation import EncoderResponseValidator
from precisionai.agritune.logging import get_logger
from precisionai.agritune.schemas.features import EncoderFeatures, concatenate_encoder_features

logger = get_logger(__name__)


@dataclass
class GatewayConfig:
    """Configuration for :class:`EncoderGateway`.

    Attributes
    ----------
    max_batch_images : int
        Maximum images per underlying encoder request.
    max_concurrency : int
        Maximum encoder requests in flight at once.
    requests_per_minute : float | None
        Sustained request-rate cap; ``None`` disables it.
    images_per_minute : float | None
        Sustained image-throughput cap; ``None`` disables it.
    retry : RetryPolicyConfig
        Retry/backoff configuration.
    """

    max_batch_images: int = 16
    max_concurrency: int = 8
    requests_per_minute: float | None = None
    images_per_minute: float | None = None
    retry: RetryPolicyConfig = field(default_factory=RetryPolicyConfig)


@dataclass
class GatewayMetrics:
    """Observability counters accumulated across every request the gateway has made.

    Attributes
    ----------
    requests_total : int
        Total encoder-request attempts made (including retries).
    requests_failed : int
        Requests (batches) that failed permanently, after exhausting retries.
    requests_retried : int
        Individual attempts that failed but were retried.
    rate_limited_count : int
        Attempts that failed with HTTP 429.
    timeout_count : int
        Attempts that failed with a timeout.
    images_encoded : int
        Total images successfully encoded.
    total_request_seconds : float
        Wall-clock time spent inside encoder calls (including retries and rate-limit waits).
    """

    requests_total: int = 0
    requests_failed: int = 0
    requests_retried: int = 0
    rate_limited_count: int = 0
    timeout_count: int = 0
    images_encoded: int = 0
    total_request_seconds: float = 0.0

    @property
    def images_per_second(self) -> float:
        """Return observed throughput, or ``0.0`` if no time has been recorded yet."""
        if self.total_request_seconds <= 0:
            return 0.0
        return self.images_encoded / self.total_request_seconds


class EncoderGateway:
    """Batches, rate-limits, retries, and validates calls to an underlying :class:`EncoderBackend`.

    Parameters
    ----------
    backend : EncoderBackend
        The backend to call (typically :class:`~precisionai.agritune.encoder.remote.RemoteEncoderBackend`
        or :class:`~precisionai.agritune.encoder.fake.FakeEncoderBackend`).
    config : GatewayConfig | None, optional
        Batching/concurrency/rate-limit/retry configuration.
    rate_limiter : RateLimiter | None, optional
        Override the rate limiter constructed from ``config`` (e.g. to share one limiter across
        multiple gateways in a distributed setting — see Phase 17).
    retry_policy : RetryPolicy | None, optional
        Override the retry policy constructed from ``config``.
    """

    def __init__(
        self,
        backend: EncoderBackend,
        config: GatewayConfig | None = None,
        *,
        rate_limiter: RateLimiter | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self._backend = backend
        self._config = config or GatewayConfig()
        self._rate_limiter = rate_limiter or RateLimiter(
            RateLimiterConfig(
                max_concurrency=self._config.max_concurrency,
                requests_per_minute=self._config.requests_per_minute,
                images_per_minute=self._config.images_per_minute,
            )
        )
        self._retry_policy = retry_policy or RetryPolicy(self._config.retry)
        self._validator = EncoderResponseValidator()
        self.metrics = GatewayMetrics()

    async def encode(self, images: Sequence[ImageInput]) -> EncoderFeatures:
        """Encode a (possibly large) batch of images through the full gateway pipeline.

        Parameters
        ----------
        images : Sequence[ImageInput]
            Images to encode; split internally into batches of at most
            ``config.max_batch_images``.

        Returns
        -------
        EncoderFeatures
            Batched features for every input image, in order.

        Raises
        ------
        ValueError
            If ``images`` is empty.
        RetryExhaustedError
            If any batch fails on every retry attempt.
        EncoderConsistencyError
            If a later batch's observed dimensions differ from the first batch seen by this
            gateway instance.
        """
        if not images:
            raise ValueError("images must be non-empty")

        batches = list(batch_items(images, max_batch_size=self._config.max_batch_images))
        logger.debug("encoding %d image(s) across %d gateway batch(es)", len(images), len(batches))
        results = await asyncio.gather(*(self._encode_batch(batch) for batch in batches))
        return concatenate_encoder_features(results)

    async def _encode_batch(self, batch: list[ImageInput]) -> EncoderFeatures:
        logger.debug("dispatching encoder batch of %d image(s)", len(batch))
        async with self._rate_limiter.acquire(num_images=len(batch)):
            start = time.monotonic()
            try:
                features = await self._retry_policy.run(
                    lambda: self._backend.encode(batch),
                    on_attempt_start=self._on_attempt_start,
                    on_attempt_failure=self._on_attempt_failure,
                )
            except EncoderError:
                self.metrics.requests_failed += 1
                raise
            finally:
                self.metrics.total_request_seconds += time.monotonic() - start

        self._validator.validate(features)
        self.metrics.images_encoded += len(batch)
        return features

    def _on_attempt_start(self, attempt: int) -> None:
        del attempt
        self.metrics.requests_total += 1

    def _on_attempt_failure(self, error: EncoderError, attempt: int, will_retry: bool) -> None:
        if isinstance(error, EncoderRateLimitError):
            self.metrics.rate_limited_count += 1
        elif isinstance(error, EncoderTimeoutError):
            self.metrics.timeout_count += 1
        if will_retry:
            self.metrics.requests_retried += 1
            logger.warning(
                "encoder request failed on attempt %d (%s: %s); retrying with backoff",
                attempt,
                type(error).__name__,
                error,
            )
        else:
            logger.warning(
                "encoder request failed on attempt %d (%s: %s); no attempts remaining",
                attempt,
                type(error).__name__,
                error,
            )
