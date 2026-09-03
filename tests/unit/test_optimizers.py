# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.optimization.optimizers."""

import torch
from torch import nn
from torch.optim import SGD, Adam, AdamW

from precisionai.agritune.optimization.optimizers import OptimizerConfig, build_optimizer


def _model() -> nn.Linear:
    # Concretely nn.Linear (not nn.Module): nn.Module's generic __getattr__ fallback types
    # dynamic attribute access (e.g. `.weight`) as `Tensor | Module`, which the `.clone()`/
    # `torch.equal()` calls below need narrowed to plain `Tensor`.
    return nn.Linear(4, 2)


def test_build_adamw_is_default() -> None:
    optimizer = build_optimizer(_model().parameters(), OptimizerConfig())
    assert isinstance(optimizer, AdamW)


def test_build_adam() -> None:
    optimizer = build_optimizer(_model().parameters(), OptimizerConfig(name="adam"))
    assert isinstance(optimizer, Adam)


def test_build_sgd_uses_momentum() -> None:
    optimizer = build_optimizer(_model().parameters(), OptimizerConfig(name="sgd", momentum=0.8))
    assert isinstance(optimizer, SGD)
    assert optimizer.param_groups[0]["momentum"] == 0.8


def test_lr_is_applied() -> None:
    optimizer = build_optimizer(_model().parameters(), OptimizerConfig(lr=1e-2))
    assert optimizer.param_groups[0]["lr"] == 1e-2


def test_weight_decay_is_applied() -> None:
    optimizer = build_optimizer(_model().parameters(), OptimizerConfig(weight_decay=0.05))
    assert optimizer.param_groups[0]["weight_decay"] == 0.05


def test_optimizer_step_updates_parameters() -> None:
    model = _model()
    optimizer = build_optimizer(model.parameters(), OptimizerConfig(lr=1.0))
    before = model.weight.clone()

    output = model(torch.randn(3, 4))
    output.sum().backward()
    optimizer.step()

    assert not torch.equal(before, model.weight)
