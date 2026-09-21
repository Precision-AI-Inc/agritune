# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Encoder benchmarking — powers ``agritune encoder benchmark``.

Empirically measures achievable throughput/latency for different ``(batch_size, concurrency)``
combinations against a live (or fake) encoder, so rate-limit settings come from data rather than
guesses. Benchmarks the raw
:class:`~precisionai.agritune.schemas.protocols.EncoderBackend` directly (not a gateway-wrapped
one) since the whole point is to discover good gateway settings.
"""

import asyncio
import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from precisionai.agritune.encoder.errors import EncoderError
from precisionai.agritune.logging import get_logger, progress_iter
from precisionai.agritune.schemas.protocols import EncoderBackend, ImageInput

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Throughput/latency observed for one ``(batch_size, concurrency)`` combination.

    Attributes
    ----------
    batch_size : int
    concurrency : int
    images_per_second : float
        Successfully encoded images divided by wall-clock time for this combination.
    latency_p50_seconds : float
    latency_p95_seconds : float
    latency_p99_seconds : float
        Percentiles over successful requests' latencies; ``0.0`` if every request failed.
    error_count : int
        Requests that raised an :class:`~precisionai.agritune.encoder.errors.EncoderError`.
    sample_error : str | None
        Message from the first failing request in this combination, prefixed with its HTTP
        status code when one is available; ``None`` when ``error_count`` is ``0``.
    """

    batch_size: int
    concurrency: int
    images_per_second: float
    latency_p50_seconds: float
    latency_p95_seconds: float
    latency_p99_seconds: float
    error_count: int
    sample_error: str | None = None


@dataclass
class BenchmarkReport:
    """Every combination measured by one :func:`run_benchmark` call.

    Attributes
    ----------
    results : list[BenchmarkResult]
    """

    results: list[BenchmarkResult] = field(default_factory=list)

    @property
    def best(self) -> BenchmarkResult | None:
        """The error-free combination with the highest throughput, or ``None`` if all errored."""
        candidates = [result for result in self.results if result.error_count == 0]
        if not candidates:
            return None
        return max(candidates, key=lambda result: result.images_per_second)


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(int(len(sorted_values) * fraction), len(sorted_values) - 1)
    return sorted_values[index]


def _format_error(exc: EncoderError) -> str:
    message = " ".join(str(exc).split())[:200]
    return f"{exc.status_code}: {message}" if exc.status_code is not None else message


async def _run_combination(
    encoder: EncoderBackend,
    *,
    batch_size: int,
    concurrency: int,
    num_requests: int,
    image_factory: Callable[[], ImageInput],
) -> BenchmarkResult:
    latencies: list[float] = []
    errors = 0
    sample_error: str | None = None
    semaphore = asyncio.Semaphore(concurrency)

    async def one_request() -> None:
        nonlocal errors, sample_error
        async with semaphore:
            images = [image_factory() for _ in range(batch_size)]
            start = time.monotonic()
            try:
                await encoder.encode(images)
            except EncoderError as exc:
                errors += 1
                if sample_error is None:
                    sample_error = _format_error(exc)
                return
            latencies.append(time.monotonic() - start)

    start_all = time.monotonic()
    await asyncio.gather(*(one_request() for _ in range(num_requests)))
    elapsed = time.monotonic() - start_all

    images_encoded = (num_requests - errors) * batch_size
    images_per_second = images_encoded / elapsed if elapsed > 0 else 0.0
    sorted_latencies = sorted(latencies)

    return BenchmarkResult(
        batch_size=batch_size,
        concurrency=concurrency,
        images_per_second=images_per_second,
        latency_p50_seconds=_percentile(sorted_latencies, 0.50),
        latency_p95_seconds=_percentile(sorted_latencies, 0.95),
        latency_p99_seconds=_percentile(sorted_latencies, 0.99),
        error_count=errors,
        sample_error=sample_error,
    )


async def run_benchmark(
    encoder: EncoderBackend,
    *,
    batch_sizes: list[int],
    concurrencies: list[int],
    image_factory: Callable[[], ImageInput],
    num_requests_per_combination: int = 10,
    show_progress: bool = False,
) -> BenchmarkReport:
    """Measure throughput/latency for every ``(batch_size, concurrency)`` combination.

    Parameters
    ----------
    encoder : EncoderBackend
        The raw backend to benchmark (typically
        :class:`~precisionai.agritune.encoder.remote.RemoteEncoderBackend`).
    batch_sizes : list[int]
        Batch sizes to test, e.g. ``[1, 4, 8, 16]``.
    concurrencies : list[int]
        Concurrency levels to test, e.g. ``[1, 2, 4, 8]``.
    image_factory : Callable[[], ImageInput]
        Produces one dummy image per call, in whatever representation ``encoder`` accepts.
    num_requests_per_combination : int, optional
        Requests issued per combination — more requests give more stable percentiles.
    show_progress : bool, optional
        Render a ``tqdm`` bar over the ``(batch_size, concurrency)`` combinations. Defaults to
        ``False`` so headless callers (e.g. the API) see no terminal output.

    Returns
    -------
    BenchmarkReport
    """
    if not batch_sizes or any(batch_size < 1 for batch_size in batch_sizes):
        raise ValueError("batch_sizes must contain only positive integers")
    if not concurrencies or any(concurrency < 1 for concurrency in concurrencies):
        raise ValueError("concurrencies must contain only positive integers")
    if num_requests_per_combination < 1:
        raise ValueError("num_requests_per_combination must be positive")

    combinations = list(itertools.product(batch_sizes, concurrencies))
    logger.info(
        "benchmarking %d combination(s) of batch_sizes=%s concurrencies=%s",
        len(combinations),
        batch_sizes,
        concurrencies,
    )
    results = []
    for batch_size, concurrency in progress_iter(
        combinations, desc="benchmark", unit="combo", disable=not show_progress
    ):
        result = await _run_combination(
            encoder,
            batch_size=batch_size,
            concurrency=concurrency,
            num_requests=num_requests_per_combination,
            image_factory=image_factory,
        )
        logger.debug(
            "batch_size=%d concurrency=%d -> %.2f img/s (errors=%d)",
            batch_size,
            concurrency,
            result.images_per_second,
            result.error_count,
        )
        results.append(result)
    return BenchmarkReport(results=results)
