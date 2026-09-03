# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Splits a sequence of images into fixed-size batches for the encoder gateway."""

from collections.abc import Iterator, Sequence
from typing import TypeVar

from precisionai.agritune.logging import get_logger

T = TypeVar("T")
logger = get_logger(__name__)


def batch_items(items: Sequence[T], *, max_batch_size: int) -> Iterator[list[T]]:
    """Yield ``items`` split into consecutive chunks of at most ``max_batch_size``.

    Parameters
    ----------
    items : Sequence[T]
        Items to split, in order.
    max_batch_size : int
        Maximum items per batch. Must be positive.

    Yields
    ------
    list[T]
        Consecutive batches; the final batch may be smaller than ``max_batch_size``. Yields
        nothing if ``items`` is empty.

    Raises
    ------
    ValueError
        If ``max_batch_size`` is not positive.
    """
    if max_batch_size <= 0:
        raise ValueError(f"max_batch_size must be positive; got {max_batch_size}")
    for start in range(0, len(items), max_batch_size):
        chunk = list(items[start : start + max_batch_size])
        logger.debug("yielding batch of %d item(s) starting at index %d", len(chunk), start)
        yield chunk
