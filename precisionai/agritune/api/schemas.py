# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Request/response models for the API layer.

Kept separate from the route modules so the wire contract is easy to read in one place. Every
field mirrors the equivalent CLI argument one-to-one — see ``precisionai.agritune.cli.main``.
"""

from pydantic import BaseModel, Field


class EncoderSelection(BaseModel):
    """Which encoder backend to use — mirrors ``_add_encoder_selection_arguments`` in the CLI."""

    base_url: str | None = Field(default=None, description="Hosted encoder API base URL; omit to use the fake encoder.")
    api_key: str | None = Field(default=None, description="Encoder API key; ignored when base_url is omitted.")
    model: str = Field(default="pai-embedding", description="Encoder model alias.")
    preprocessing: str = Field(
        default="", description="A stable label for preprocessing params, for cache invalidation."
    )


class EncoderFingerprintRequest(BaseModel):
    """Identifies the encoder configuration a feature store was already built with."""

    model: str = Field(default="fake-encoder", description="Encoder model alias the feature store was built with.")
    revision: str | None = Field(default="fake-v1", description="Encoder revision the feature store was built with.")
    preprocessing: str = Field(default="", description="Preprocessing label used when the store was built.")


class ValidationIssueResponse(BaseModel):
    """One problem found while validating a manifest."""

    sample_id: str | None
    category: str
    message: str


class DatasetValidateRequest(BaseModel):
    """Body for ``POST /dataset/validate``."""

    manifest_path: str
    num_classes: int | None = None
    ignore_index: int | None = None


class DatasetValidateResponse(BaseModel):
    """Response for ``POST /dataset/validate``."""

    is_valid: bool
    issues: list[ValidationIssueResponse]


class DatasetInspectResponse(BaseModel):
    """Response for ``GET /dataset/inspect``."""

    num_samples: int
    metadata_columns: list[str]
    class_pixel_counts: dict[int, int]


class FeaturesBuildRequest(BaseModel):
    """Body for ``POST /features/build``."""

    manifest_path: str
    store: str
    encoder: EncoderSelection = Field(default_factory=EncoderSelection)


class FeaturesBuildResponse(BaseModel):
    """Response for ``POST /features/build``."""

    total: int
    computed: int
    skipped: int
    failed: int
    failed_sample_ids: list[str]


class FeaturesStoreRequest(BaseModel):
    """Body for ``POST /features/verify`` and ``POST /features/clean``."""

    store: str


class FeaturesVerifyResponse(BaseModel):
    """Response for ``POST /features/verify``."""

    is_valid: bool
    total: int
    corrupted_keys: list[str]


class FeaturesInspectResponse(BaseModel):
    """Response for ``GET /features/inspect``."""

    total_entries: int
    encoder_models: dict[str, int]
    patch_dims: dict[int, int]


class FeaturesCleanResponse(BaseModel):
    """Response for ``POST /features/clean``."""

    removed: list[str]


class TrainRequest(BaseModel):
    """Body for ``POST /train``."""

    config_path: str
    overrides: list[str] = Field(default_factory=list)


class TrainResponse(BaseModel):
    """Response for ``POST /train``."""

    run_directory: str
    final_epoch: int
    global_optimizer_step: int
    best_metric: float | None
    val_metrics: dict[str, float]


class ScoredRunRequest(BaseModel):
    """Shared body fields for ``POST /evaluate`` and ``POST /predict``."""

    manifest_path: str
    store: str
    checkpoint_path: str
    num_classes: int
    decoder: str = Field(default="linear", description='"linear" or "token_fpn".')
    batch_size: int = 4
    sample_ids: list[str] | None = None
    encoder_fingerprint: EncoderFingerprintRequest = Field(default_factory=EncoderFingerprintRequest)


class EvaluateResponse(BaseModel):
    """Response for ``POST /evaluate``."""

    metrics: dict[str, float]


class PredictRequest(ScoredRunRequest):
    """Body for ``POST /predict``."""

    output_dir: str
    overlays: bool = False
    overlay_alpha: float = 0.5


class PredictResponse(BaseModel):
    """Response for ``POST /predict``."""

    written: list[str]


class EncoderBenchmarkRequest(BaseModel):
    """Body for ``POST /encoder/benchmark``."""

    encoder: EncoderSelection = Field(default_factory=EncoderSelection)
    batch_sizes: list[int] = Field(default_factory=lambda: [1, 4, 8, 16])
    concurrencies: list[int] = Field(default_factory=lambda: [1, 2, 4, 8])
    num_requests: int = 10
    image_size: int = 64


class BenchmarkResultResponse(BaseModel):
    """Throughput/latency observed for one ``(batch_size, concurrency)`` combination."""

    batch_size: int
    concurrency: int
    images_per_second: float
    latency_p50_seconds: float
    latency_p95_seconds: float
    latency_p99_seconds: float
    error_count: int


class EncoderBenchmarkResponse(BaseModel):
    """Response for ``POST /encoder/benchmark``."""

    results: list[BenchmarkResultResponse]
    best: BenchmarkResultResponse | None


class ErrorResponse(BaseModel):
    """Body of every non-2xx response — see ``precisionai.agritune.api.app``'s exception handlers."""

    detail: str
