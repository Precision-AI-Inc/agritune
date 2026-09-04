# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.services.benchmark_service."""

import pytest

from precisionai.agritune.encoder.fake import FakeEncoderBackend, FakeEncoderConfig
from precisionai.agritune.services.benchmark_service import BenchmarkReport, BenchmarkResult, run_benchmark


def _image_factory() -> str:
    return "dummy-image"


async def test_run_benchmark_reports_plausible_throughput_and_latency() -> None:
    encoder = FakeEncoderBackend(FakeEncoderConfig(latency_seconds=0.01))
    report = await run_benchmark(
        encoder,
        batch_sizes=[2],
        concurrencies=[1],
        image_factory=_image_factory,
        num_requests_per_combination=4,
    )

    assert len(report.results) == 1
    result = report.results[0]
    assert result.batch_size == 2
    assert result.concurrency == 1
    assert result.error_count == 0
    assert result.images_per_second > 0
    assert result.latency_p50_seconds >= 0.01
    assert result.latency_p95_seconds >= result.latency_p50_seconds


async def test_run_benchmark_with_show_progress_still_covers_every_combination() -> None:
    encoder = FakeEncoderBackend()
    report = await run_benchmark(
        encoder,
        batch_sizes=[1, 2],
        concurrencies=[1],
        image_factory=_image_factory,
        num_requests_per_combination=2,
        show_progress=True,
    )

    assert len(report.results) == 2


async def test_run_benchmark_covers_every_combination() -> None:
    encoder = FakeEncoderBackend()
    report = await run_benchmark(
        encoder,
        batch_sizes=[1, 4],
        concurrencies=[1, 2],
        image_factory=_image_factory,
        num_requests_per_combination=2,
    )
    combos = {(result.batch_size, result.concurrency) for result in report.results}
    assert combos == {(1, 1), (1, 2), (4, 1), (4, 2)}


async def test_run_benchmark_records_errors_and_zero_throughput() -> None:
    encoder = FakeEncoderBackend(FakeEncoderConfig(failure_probability=1.0, status_code_on_failure=500))
    report = await run_benchmark(
        encoder,
        batch_sizes=[1],
        concurrencies=[1],
        image_factory=_image_factory,
        num_requests_per_combination=3,
    )
    result = report.results[0]
    assert result.error_count == 3
    assert result.images_per_second == 0.0
    assert result.latency_p50_seconds == 0.0


async def test_run_benchmark_records_sample_error_message() -> None:
    encoder = FakeEncoderBackend(FakeEncoderConfig(failure_probability=1.0, status_code_on_failure=503))
    report = await run_benchmark(
        encoder,
        batch_sizes=[1],
        concurrencies=[1],
        image_factory=_image_factory,
        num_requests_per_combination=2,
    )
    result = report.results[0]
    assert result.sample_error is not None
    assert result.sample_error.startswith("503:")
    assert "simulated 503" in result.sample_error


async def test_run_benchmark_sample_error_is_none_on_success() -> None:
    encoder = FakeEncoderBackend()
    report = await run_benchmark(
        encoder,
        batch_sizes=[1],
        concurrencies=[1],
        image_factory=_image_factory,
        num_requests_per_combination=2,
    )
    assert report.results[0].sample_error is None


@pytest.mark.parametrize(
    ("batch_sizes", "concurrencies", "num_requests", "message"),
    [
        ([], [1], 1, "batch_sizes"),
        ([0], [1], 1, "batch_sizes"),
        ([1], [], 1, "concurrencies"),
        ([1], [0], 1, "concurrencies"),
        ([1], [1], 0, "num_requests_per_combination"),
    ],
)
async def test_run_benchmark_rejects_nonpositive_workload_dimensions(
    batch_sizes: list[int], concurrencies: list[int], num_requests: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        await run_benchmark(
            FakeEncoderBackend(),
            batch_sizes=batch_sizes,
            concurrencies=concurrencies,
            image_factory=_image_factory,
            num_requests_per_combination=num_requests,
        )


def test_benchmark_report_best_returns_none_when_all_errored() -> None:
    report = BenchmarkReport(
        results=[
            BenchmarkResult(
                batch_size=1,
                concurrency=1,
                images_per_second=0.0,
                latency_p50_seconds=0.0,
                latency_p95_seconds=0.0,
                latency_p99_seconds=0.0,
                error_count=5,
            )
        ]
    )
    assert report.best is None


def test_benchmark_report_best_picks_highest_throughput() -> None:
    slow = BenchmarkResult(
        batch_size=1,
        concurrency=1,
        images_per_second=10.0,
        latency_p50_seconds=0.1,
        latency_p95_seconds=0.1,
        latency_p99_seconds=0.1,
        error_count=0,
    )
    fast = BenchmarkResult(
        batch_size=8,
        concurrency=4,
        images_per_second=50.0,
        latency_p50_seconds=0.2,
        latency_p95_seconds=0.2,
        latency_p99_seconds=0.2,
        error_count=0,
    )
    report = BenchmarkReport(results=[slow, fast])
    assert report.best is fast


def test_benchmark_report_empty_results_best_is_none() -> None:
    assert BenchmarkReport(results=[]).best is None
