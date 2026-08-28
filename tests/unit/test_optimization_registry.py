# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.optimization.registry."""

from precisionai.agritune.optimization.registry import OPTIMIZER_NAMES, SCHEDULER_NAMES


def test_optimizer_names_include_all_supported_optimizers() -> None:
    assert set(OPTIMIZER_NAMES) == {"adamw", "adam", "sgd"}


def test_scheduler_names_include_all_supported_schedulers() -> None:
    assert set(SCHEDULER_NAMES) == {"constant", "cosine", "linear_warmup", "polynomial", "plateau"}
