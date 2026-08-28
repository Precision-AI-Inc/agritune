# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Training progress state and RNG capture/restore, for exact checkpoint resume.

Tracks ``epoch``, ``micro_step``, and ``global_optimizer_step`` separately — schedulers step on
the optimizer step, not the micro-batch, under gradient accumulation (see
``agritune_implementation_plan.md`` §12).
"""

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class TrainingState:
    """Mutable training progress, checkpointed alongside model/optimizer state.

    Attributes
    ----------
    epoch : int
        The next epoch to run (0-indexed) — incremented only once an epoch fully completes, so
        resuming from a checkpoint continues at the right epoch rather than repeating or skipping
        one.
    micro_step : int
        Total forward/backward passes so far, including those absorbed by gradient accumulation.
    global_optimizer_step : int
        Total ``optimizer.step()`` calls so far — what schedulers should be driven by.
    best_metric : float | None
        Best validation metric seen so far, or ``None`` before the first evaluation.
    best_metric_name : str | None
        Which metric ``best_metric`` refers to.
    """

    epoch: int = 0
    micro_step: int = 0
    global_optimizer_step: int = 0
    best_metric: float | None = None
    best_metric_name: str | None = None


def set_deterministic_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU and, if available, all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover — CI runs CPU-only
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    """Capture Python/NumPy/PyTorch (CPU + CUDA, if available) RNG state for checkpointing.

    Returns
    -------
    dict[str, Any]
        Pass to :func:`restore_rng_state` to resume with bit-identical randomness.
    """
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():  # pragma: no cover — CI runs CPU-only
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG state previously captured by :func:`capture_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():  # pragma: no cover — CI runs CPU-only
        torch.cuda.set_rng_state_all(state["torch_cuda"])
