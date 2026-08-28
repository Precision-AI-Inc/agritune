# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.training.precision.PrecisionContext."""

import torch
from torch import nn

from precisionai.agritune.training.precision import PrecisionConfig, PrecisionContext


def test_fp32_autocast_is_a_no_op() -> None:
    context = PrecisionContext(PrecisionConfig(mode="fp32"))
    model = nn.Linear(4, 2)
    with context.autocast():
        output = model(torch.randn(3, 4))
    assert output.dtype == torch.float32


def test_bf16_autocast_produces_bf16_output() -> None:
    context = PrecisionContext(PrecisionConfig(mode="bf16"), device_type="cpu")
    model = nn.Linear(4, 2)
    with context.autocast():
        output = model(torch.randn(3, 4))
    assert output.dtype == torch.bfloat16


def test_fp16_autocast_produces_fp16_output() -> None:
    context = PrecisionContext(PrecisionConfig(mode="fp16"), device_type="cpu")
    model = nn.Linear(4, 2)
    with context.autocast():
        output = model(torch.randn(3, 4))
    assert output.dtype == torch.float16


def test_fp16_scaler_is_disabled_on_cpu() -> None:
    context = PrecisionContext(PrecisionConfig(mode="fp16"), device_type="cpu")
    assert context.scaler.is_enabled() is False


def test_backward_and_step_update_parameters() -> None:
    context = PrecisionContext(PrecisionConfig(mode="fp32"))
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    before = model.weight.clone()

    with context.autocast():
        loss = model(torch.randn(3, 4)).sum()
    context.backward(loss)
    context.step(optimizer)

    assert not torch.equal(before, model.weight)


def test_scaler_state_dict_roundtrips() -> None:
    # On CPU the scaler is always disabled (see test_fp16_scaler_is_disabled_on_cpu), so its
    # state_dict is empty — this test only verifies save/load doesn't raise on that empty state.
    context = PrecisionContext(PrecisionConfig(mode="fp32"))
    state = context.state_dict()
    other = PrecisionContext(PrecisionConfig(mode="fp32"))
    other.load_state_dict(state)  # should not raise
    assert other.state_dict() == state


def test_unscale_does_not_raise_when_scaler_disabled() -> None:
    context = PrecisionContext(PrecisionConfig(mode="fp32"))
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    loss = model(torch.randn(3, 4)).sum()
    context.backward(loss)
    context.unscale_(optimizer)  # should not raise
