# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Scheduler registry.

Every scheduler here steps per **optimizer step**, not per micro-batch: gradient accumulation
tracks ``micro_step`` and ``global_optimizer_step`` separately, and schedulers must be driven by
the latter.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler, ReduceLROnPlateau

SchedulerName = Literal["constant", "cosine", "linear_warmup", "polynomial", "plateau"]


@dataclass
class SchedulerConfig:
    """Configuration for :func:`build_scheduler`.

    Attributes
    ----------
    name : SchedulerName
        Which scheduler to build.
    warmup_steps : int
        Linear warmup duration, in optimizer steps. ``0`` disables warmup. Ignored by
        ``"constant"`` and ``"plateau"``.
    total_steps : int | None
        Total optimizer steps in the run. Required by ``"cosine"``, ``"linear_warmup"``, and
        ``"polynomial"``.
    polynomial_power : float
        Decay exponent for ``"polynomial"``.
    plateau_factor : float
        Multiplicative LR reduction applied by ``"plateau"`` when triggered.
    plateau_patience : int
        Epochs with no improvement before ``"plateau"`` reduces the LR.
    """

    name: SchedulerName = "constant"
    warmup_steps: int = 0
    total_steps: int | None = None
    polynomial_power: float = 1.0
    plateau_factor: float = 0.5
    plateau_patience: int = 10


def build_scheduler(optimizer: Optimizer, config: SchedulerConfig) -> LRScheduler | ReduceLROnPlateau:
    """Construct the scheduler named in ``config`` over ``optimizer``.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer to schedule.
    config : SchedulerConfig
        Scheduler selection and hyperparameters.

    Returns
    -------
    torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau
        Call ``.step()`` once per optimizer step (``"plateau"`` instead expects
        ``.step(metric)`` once per evaluation).

    Raises
    ------
    ValueError
        If ``config.name`` is not a supported scheduler, or a step-count-dependent scheduler is
        requested without ``total_steps``.
    """
    if config.name == "constant":
        return LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    if config.name == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=config.plateau_factor, patience=config.plateau_patience)
    if config.name in ("cosine", "linear_warmup", "polynomial"):
        return LambdaLR(optimizer, lr_lambda=_step_factor_fn(config))
    raise ValueError(f"unsupported scheduler: {config.name!r}")  # pragma: no cover — exhaustive over SchedulerName


def _step_factor_fn(config: SchedulerConfig) -> Callable[[int], float]:
    if config.total_steps is None:
        raise ValueError(f"scheduler {config.name!r} requires total_steps")
    total_steps = config.total_steps
    warmup_steps = config.warmup_steps

    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
        if config.name == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if config.name == "linear_warmup":
            return max(0.0, 1.0 - progress)
        return (1.0 - progress) ** config.polynomial_power  # polynomial

    return factor
