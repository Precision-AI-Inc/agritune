# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.training.distributed.DistributedContext."""

import pytest

from precisionai.agritune.training.distributed import DistributedContext


def test_default_is_single_process_main() -> None:
    context = DistributedContext()
    assert context.rank == 0
    assert context.world_size == 1
    assert context.is_main_process is True


def test_non_zero_rank_is_not_main_process() -> None:
    context = DistributedContext(rank=1, world_size=4)
    assert context.is_main_process is False


def test_from_environment_defaults_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    context = DistributedContext.from_environment()
    assert context == DistributedContext(rank=0, world_size=1, local_rank=0)


def test_from_environment_reads_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "0")
    context = DistributedContext.from_environment()
    assert context == DistributedContext(rank=2, world_size=4, local_rank=0)
