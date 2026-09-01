# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.training.state."""

import random

import numpy as np
import torch

from precisionai.agritune.training.state import (
    TrainingState,
    capture_rng_state,
    restore_rng_state,
    set_deterministic_seed,
)


def test_set_deterministic_seed_makes_torch_rand_reproducible() -> None:
    set_deterministic_seed(42)
    first = torch.rand(5)
    set_deterministic_seed(42)
    second = torch.rand(5)
    assert torch.equal(first, second)


def test_set_deterministic_seed_affects_python_random() -> None:
    set_deterministic_seed(1)
    first = random.random()
    set_deterministic_seed(1)
    second = random.random()
    assert first == second


def test_set_deterministic_seed_affects_numpy() -> None:
    set_deterministic_seed(1)
    first = np.random.rand()
    set_deterministic_seed(1)
    second = np.random.rand()
    assert first == second


def test_capture_and_restore_rng_state_reproduces_subsequent_draws() -> None:
    set_deterministic_seed(7)
    state = capture_rng_state()
    expected_torch = torch.rand(3)
    expected_python = random.random()
    expected_numpy = np.random.rand()

    set_deterministic_seed(999)  # perturb
    torch.rand(10)

    restore_rng_state(state)
    assert torch.equal(torch.rand(3), expected_torch)
    assert random.random() == expected_python
    assert np.random.rand() == expected_numpy


def test_training_state_defaults() -> None:
    state = TrainingState()
    assert state.epoch == 0
    assert state.micro_step == 0
    assert state.batch_in_epoch == 0
    assert state.global_optimizer_step == 0
    assert state.best_metric is None
    assert state.best_metric_name is None
    assert state.epochs_without_improvement == 0
