# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Mixed-precision training support: fp32, fp16 (with gradient scaling), and bf16."""

import contextlib
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch.optim import Optimizer

PrecisionMode = Literal["fp32", "fp16", "bf16"]


@dataclass
class PrecisionConfig:
    """Configuration for :class:`PrecisionContext`.

    Attributes
    ----------
    mode : PrecisionMode
        ``"fp32"`` disables autocast entirely. ``"fp16"``/``"bf16"`` wrap the forward pass in
        ``torch.autocast``; only ``"fp16"`` uses gradient scaling (``bf16``'s wider dynamic range
        makes it unnecessary).
    """

    mode: PrecisionMode = "fp32"


class PrecisionContext:
    """Wraps ``torch.autocast`` and (for fp16) a ``torch.amp.GradScaler`` behind one interface.

    Parameters
    ----------
    config : PrecisionConfig
        Precision selection.
    device_type : str, optional
        ``"cuda"`` or ``"cpu"``. Gradient scaling is only meaningful (and enabled) for fp16 on
        CUDA — fp16 on CPU has no accelerated kernels and scaling support is unreliable there.
    """

    def __init__(self, config: PrecisionConfig, *, device_type: str = "cpu") -> None:
        self._config = config
        self._device_type = device_type
        scaler_enabled = config.mode == "fp16" and device_type == "cuda"
        self.scaler = torch.amp.GradScaler(device_type, enabled=scaler_enabled)

    def autocast(self) -> contextlib.AbstractContextManager[Any]:
        """Return the autocast context manager for the forward pass (a no-op under fp32).

        Typed to yield ``Any`` rather than ``None``: ``contextlib.nullcontext()`` (fp32) and
        ``torch.autocast`` (fp16/bf16) disagree on what ``__enter__`` returns, but no caller binds
        it (``with self.precision.autocast():``) so the yielded value is never actually used.
        """
        if self._config.mode == "fp32":
            return contextlib.nullcontext()
        dtype = torch.float16 if self._config.mode == "fp16" else torch.bfloat16
        return torch.autocast(device_type=self._device_type, dtype=dtype)

    def backward(self, loss: torch.Tensor) -> None:
        """Scale (if fp16) and back-propagate ``loss``."""
        self.scaler.scale(loss).backward()

    def step(self, optimizer: Optimizer) -> None:
        """Unscale (if fp16) and step ``optimizer``, then update the scale factor."""
        self.scaler.step(optimizer)
        self.scaler.update()

    def unscale_(self, optimizer: Optimizer) -> None:
        """Unscale gradients in-place — call before gradient clipping under fp16."""
        self.scaler.unscale_(optimizer)

    def state_dict(self) -> dict[str, Any]:
        """Return the gradient scaler's state, for checkpointing."""
        return self.scaler.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the gradient scaler's state from a checkpoint."""
        self.scaler.load_state_dict(state)
