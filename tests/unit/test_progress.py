# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.logging.progress."""

from precisionai.agritune.logging import progress_iter


def test_progress_iter_yields_items_unchanged_when_enabled() -> None:
    assert list(progress_iter([1, 2, 3], desc="items")) == [1, 2, 3]


def test_progress_iter_yields_items_unchanged_when_disabled() -> None:
    assert list(progress_iter([1, 2, 3], disable=True)) == [1, 2, 3]


def test_progress_iter_handles_iterables_without_len() -> None:
    assert list(progress_iter(iter([1, 2, 3]))) == [1, 2, 3]


def test_progress_iter_handles_empty_iterable() -> None:
    assert list(progress_iter([])) == []
