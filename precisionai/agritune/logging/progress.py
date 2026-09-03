# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""A single, shared ``tqdm`` wrapper for every progress bar inside ``precisionai``.

Every internal call site goes through :func:`progress_iter` instead of importing ``tqdm``
directly, so a future distributed-training rollout (see
``precisionai.agritune.training.distributed.DistributedContext``) only has to change ``disable=``
at each call site (e.g. ``disable=not distributed.is_main_process``) rather than hunt down
scattered ``tqdm(...)`` calls.
"""

from collections.abc import Iterable
from typing import TypeVar

from tqdm import tqdm

T = TypeVar("T")


def progress_iter(
    iterable: Iterable[T], *, desc: str | None = None, unit: str = "it", disable: bool = False
) -> Iterable[T]:
    """Wrap ``iterable`` with a ``tqdm`` progress bar.

    Parameters
    ----------
    iterable : Iterable[T]
        Items to iterate over. ``tqdm`` infers a total from ``len(iterable)`` when available and
        falls back to a counter-only display otherwise.
    desc : str | None, optional
        Short label shown before the bar.
    unit : str, optional
        Unit label for the rate/counter display (default ``"it"``).
    disable : bool, optional
        When ``True``, no bar is rendered but ``iterable`` still yields normally — the hook a
        future multi-process rank check should set.

    Returns
    -------
    Iterable[T]
        ``iterable``, wrapped for display.
    """
    return tqdm(iterable, desc=desc, unit=unit, disable=disable, leave=False)
