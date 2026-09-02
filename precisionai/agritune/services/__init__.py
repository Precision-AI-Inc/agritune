# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0
"""Orchestration layer shared by the CLI and API — the only place core subsystems are wired together."""

from precisionai.agritune.services.benchmark_service import BenchmarkReport, BenchmarkResult, run_benchmark
from precisionai.agritune.services.config_template_service import render_config_template, write_config_template
from precisionai.agritune.services.dataset_service import (
    inspect_dataset,
    render_manifest_template,
    validate_dataset,
    write_manifest_template,
)
from precisionai.agritune.services.encoder_selection import build_encoder, build_raw_encoder
from precisionai.agritune.services.evaluation_service import EvaluationRunConfig, run_evaluation
from precisionai.agritune.services.feature_service import build_features
from precisionai.agritune.services.prediction_service import PredictionRunConfig, run_prediction
from precisionai.agritune.services.tracking_selection import TrackingSelection, build_trackers
from precisionai.agritune.services.training_service import (
    AugmentationSelection,
    TrainingRunConfig,
    TrainingRunResult,
    run_training,
)

__all__ = [
    "AugmentationSelection",
    "BenchmarkReport",
    "BenchmarkResult",
    "EvaluationRunConfig",
    "PredictionRunConfig",
    "TrackingSelection",
    "TrainingRunConfig",
    "TrainingRunResult",
    "build_encoder",
    "build_features",
    "build_raw_encoder",
    "build_trackers",
    "inspect_dataset",
    "render_config_template",
    "render_manifest_template",
    "run_benchmark",
    "run_evaluation",
    "run_prediction",
    "run_training",
    "validate_dataset",
    "write_config_template",
    "write_manifest_template",
]
