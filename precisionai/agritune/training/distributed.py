# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Minimal distributed-readiness context.

Full DDP wiring is deliberately deferred — single-GPU training must be reliable first. This
module exists so the trainer and checkpointing
code can already ask "am I the main process" and "how many ranks" without hardcoding
single-process assumptions everywhere they matter (e.g. only rank 0 should write checkpoints or
log to a tracker).
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedContext:
    """Identifies this process's place in a (possibly single-process) training job.

    Attributes
    ----------
    rank : int
        Global rank, ``0`` for a single-process run.
    world_size : int
        Total number of ranks, ``1`` for a single-process run.
    local_rank : int
        Rank within this machine.
    """

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_main_process(self) -> bool:
        """Return whether this process should perform singleton work (checkpointing, tracking)."""
        return self.rank == 0

    @classmethod
    def from_environment(cls) -> "DistributedContext":
        """Build a context from the standard ``RANK``/``WORLD_SIZE``/``LOCAL_RANK`` env vars.

        Falls back to single-process defaults (rank 0, world size 1) when unset.
        """
        return cls(
            rank=int(os.environ.get("RANK", "0")),
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
            local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        )
