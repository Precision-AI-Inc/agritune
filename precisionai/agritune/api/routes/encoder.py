# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Route for ``agritune encoder benchmark`` — see ``precisionai.agritune.cli.handlers``."""

from fastapi import APIRouter
from PIL import Image

from precisionai.agritune.api.schemas import BenchmarkResultResponse, EncoderBenchmarkRequest, EncoderBenchmarkResponse
from precisionai.agritune.services.benchmark_service import BenchmarkReport, BenchmarkResult, run_benchmark
from precisionai.agritune.services.encoder_selection import build_raw_encoder

router = APIRouter(prefix="/encoder", tags=["encoder"])


def _to_response(result: BenchmarkResult) -> BenchmarkResultResponse:
    return BenchmarkResultResponse(
        batch_size=result.batch_size,
        concurrency=result.concurrency,
        images_per_second=result.images_per_second,
        latency_p50_seconds=result.latency_p50_seconds,
        latency_p95_seconds=result.latency_p95_seconds,
        latency_p99_seconds=result.latency_p99_seconds,
        error_count=result.error_count,
    )


@router.post("/benchmark", response_model=EncoderBenchmarkResponse)
async def benchmark(request: EncoderBenchmarkRequest) -> EncoderBenchmarkResponse:
    """Measure throughput/latency for every (batch size, concurrency) combination."""
    backend, _ = build_raw_encoder(
        base_url=request.encoder.base_url,
        api_key=request.encoder.api_key,
        model=request.encoder.model,
        preprocessing=request.encoder.preprocessing,
    )

    def image_factory() -> Image.Image:
        return Image.new("RGB", (request.image_size, request.image_size))

    report: BenchmarkReport = await run_benchmark(
        backend,
        batch_sizes=request.batch_sizes,
        concurrencies=request.concurrencies,
        image_factory=image_factory,
        num_requests_per_combination=request.num_requests,
    )

    return EncoderBenchmarkResponse(
        results=[_to_response(result) for result in report.results],
        best=_to_response(report.best) if report.best is not None else None,
    )
