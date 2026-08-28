# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.encoder.batching."""

import pytest

from precisionai.agritune.encoder.batching import batch_items


def test_batch_items_splits_evenly() -> None:
    batches = list(batch_items([1, 2, 3, 4], max_batch_size=2))
    assert batches == [[1, 2], [3, 4]]


def test_batch_items_final_batch_is_smaller() -> None:
    batches = list(batch_items([1, 2, 3, 4, 5], max_batch_size=2))
    assert batches == [[1, 2], [3, 4], [5]]


def test_batch_items_single_batch_when_smaller_than_max() -> None:
    batches = list(batch_items([1, 2], max_batch_size=10))
    assert batches == [[1, 2]]


def test_batch_items_empty_input_yields_nothing() -> None:
    assert list(batch_items([], max_batch_size=4)) == []


def test_batch_items_rejects_non_positive_max_batch_size() -> None:
    with pytest.raises(ValueError, match="max_batch_size must be positive"):
        list(batch_items([1], max_batch_size=0))
