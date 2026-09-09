# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Optimizer registry. Default: AdamW."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Literal

from torch import Tensor
from torch.optim import SGD, Adam, AdamW, Optimizer

OptimizerName = Literal["adamw", "adam", "sgd"]


@dataclass
class OptimizerConfig:
    """Configuration for :func:`build_optimizer`.

    Attributes
    ----------
    name : OptimizerName
        Which optimizer to build.
    lr : float
        Learning rate.
    weight_decay : float
        Weight decay (L2 penalty for SGD; decoupled for Adam/AdamW).
    betas : tuple[float, float]
        Adam/AdamW momentum coefficients. Ignored for SGD.
    momentum : float
        SGD momentum. Ignored for Adam/AdamW.
    """

    name: OptimizerName = "adamw"
    lr: float = 3e-4
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.999)
    momentum: float = 0.9


def build_optimizer(parameters: Iterable[Tensor] | Iterator[Tensor], config: OptimizerConfig) -> Optimizer:
    """Construct the optimizer named in ``config`` over ``parameters``.

    Parameters
    ----------
    parameters : Iterable[torch.Tensor]
        Typically ``model.parameters()``.
    config : OptimizerConfig
        Optimizer selection and hyperparameters.

    Returns
    -------
    torch.optim.Optimizer

    Raises
    ------
    ValueError
        If ``config.name`` is not a supported optimizer.
    """
    if config.name == "adamw":
        return AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay, betas=config.betas)
    if config.name == "adam":
        return Adam(parameters, lr=config.lr, weight_decay=config.weight_decay, betas=config.betas)
    if config.name == "sgd":
        return SGD(parameters, lr=config.lr, weight_decay=config.weight_decay, momentum=config.momentum)
    raise ValueError(f"unsupported optimizer: {config.name!r}")  # pragma: no cover — exhaustive over OptimizerName
