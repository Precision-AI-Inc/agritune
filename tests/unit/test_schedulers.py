# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.optimization.schedulers."""

import pytest
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from precisionai.agritune.optimization.schedulers import SchedulerConfig, build_scheduler

_BASE_LR = 1.0


def _optimizer() -> SGD:
    return SGD(nn.Linear(2, 2).parameters(), lr=_BASE_LR)


def _lrs_over_steps(optimizer: SGD, scheduler: LRScheduler, num_steps: int) -> list[float]:
    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(num_steps):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    return lrs


def test_constant_scheduler_never_changes_lr() -> None:
    optimizer = _optimizer()
    scheduler = build_scheduler(optimizer, SchedulerConfig(name="constant"))
    lrs = _lrs_over_steps(optimizer, scheduler, 5)
    assert all(lr == pytest.approx(_BASE_LR) for lr in lrs)


def test_cosine_decays_to_near_zero_at_total_steps() -> None:
    optimizer = _optimizer()
    scheduler = build_scheduler(optimizer, SchedulerConfig(name="cosine", total_steps=10))
    lrs = _lrs_over_steps(optimizer, scheduler, 10)
    assert lrs[0] == pytest.approx(_BASE_LR)
    assert lrs[-1] == pytest.approx(0.0, abs=1e-6)


def test_cosine_with_warmup_ramps_up_first() -> None:
    optimizer = _optimizer()
    scheduler = build_scheduler(optimizer, SchedulerConfig(name="cosine", warmup_steps=4, total_steps=10))
    lrs = _lrs_over_steps(optimizer, scheduler, 4)
    assert lrs[0] < lrs[1] < lrs[2] < lrs[3]


def test_linear_warmup_decays_linearly_to_zero() -> None:
    optimizer = _optimizer()
    scheduler = build_scheduler(optimizer, SchedulerConfig(name="linear_warmup", total_steps=4))
    lrs = _lrs_over_steps(optimizer, scheduler, 4)
    assert lrs == pytest.approx([1.0, 0.75, 0.5, 0.25, 0.0])


def test_polynomial_decay_matches_formula() -> None:
    optimizer = _optimizer()
    scheduler = build_scheduler(optimizer, SchedulerConfig(name="polynomial", total_steps=4, polynomial_power=2.0))
    lrs = _lrs_over_steps(optimizer, scheduler, 4)
    assert lrs == pytest.approx([1.0, 0.5625, 0.25, 0.0625, 0.0])


def test_missing_total_steps_raises_for_cosine() -> None:
    optimizer = _optimizer()
    with pytest.raises(ValueError, match="requires total_steps"):
        build_scheduler(optimizer, SchedulerConfig(name="cosine"))


def test_missing_total_steps_raises_for_linear_warmup() -> None:
    optimizer = _optimizer()
    with pytest.raises(ValueError, match="requires total_steps"):
        build_scheduler(optimizer, SchedulerConfig(name="linear_warmup"))


def test_missing_total_steps_raises_for_polynomial() -> None:
    optimizer = _optimizer()
    with pytest.raises(ValueError, match="requires total_steps"):
        build_scheduler(optimizer, SchedulerConfig(name="polynomial"))


def test_plateau_scheduler_reduces_lr_when_metric_stalls() -> None:
    optimizer = _optimizer()
    scheduler = build_scheduler(optimizer, SchedulerConfig(name="plateau", plateau_patience=1, plateau_factor=0.5))
    assert isinstance(scheduler, ReduceLROnPlateau)

    scheduler.step(1.0)
    scheduler.step(1.0)  # no improvement, patience=1 exceeded
    scheduler.step(1.0)
    assert optimizer.param_groups[0]["lr"] < _BASE_LR
