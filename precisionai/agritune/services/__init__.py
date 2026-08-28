# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0
"""Orchestration layer shared by the CLI and API — the only place core subsystems are wired together."""

from precisionai.agritune.services.benchmark_service import BenchmarkReport, BenchmarkResult, run_benchmark
from precisionai.agritune.services.dataset_service import inspect_dataset, validate_dataset
from precisionai.agritune.services.evaluation_service import EvaluationRunConfig, run_evaluation
from precisionai.agritune.services.feature_service import build_features
from precisionai.agritune.services.prediction_service import PredictionRunConfig, run_prediction
from precisionai.agritune.services.training_service import TrainingRunConfig, TrainingRunResult, run_training

__all__ = [
    "BenchmarkReport",
    "BenchmarkResult",
    "EvaluationRunConfig",
    "PredictionRunConfig",
    "TrainingRunConfig",
    "TrainingRunResult",
    "build_features",
    "inspect_dataset",
    "run_benchmark",
    "run_evaluation",
    "run_prediction",
    "run_training",
    "validate_dataset",
]
