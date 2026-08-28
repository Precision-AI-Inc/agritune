# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.augmentations.image.seeding."""

import pytest

from precisionai.agritune.augmentations.image.seeding import (
    derive_hybrid_seed,
    derive_offline_seed,
    derive_online_seed,
)


def test_offline_seed_is_deterministic() -> None:
    first = derive_offline_seed(global_seed=1, sample_id="s1", variant=0)
    second = derive_offline_seed(global_seed=1, sample_id="s1", variant=0)
    assert first == second


def test_offline_seed_differs_by_variant() -> None:
    seed_0 = derive_offline_seed(global_seed=1, sample_id="s1", variant=0)
    seed_1 = derive_offline_seed(global_seed=1, sample_id="s1", variant=1)
    assert seed_0 != seed_1


def test_offline_seed_differs_by_sample_id() -> None:
    seed_a = derive_offline_seed(global_seed=1, sample_id="a", variant=0)
    seed_b = derive_offline_seed(global_seed=1, sample_id="b", variant=0)
    assert seed_a != seed_b


def test_online_seed_is_deterministic() -> None:
    first = derive_online_seed(global_seed=1, epoch=0, sample_id="s1", occurrence=0)
    second = derive_online_seed(global_seed=1, epoch=0, sample_id="s1", occurrence=0)
    assert first == second


def test_online_seed_differs_by_epoch() -> None:
    seed_epoch_0 = derive_online_seed(global_seed=1, epoch=0, sample_id="s1", occurrence=0)
    seed_epoch_1 = derive_online_seed(global_seed=1, epoch=1, sample_id="s1", occurrence=0)
    assert seed_epoch_0 != seed_epoch_1


def test_online_seed_differs_by_occurrence() -> None:
    first = derive_online_seed(global_seed=1, epoch=0, sample_id="s1", occurrence=0)
    second = derive_online_seed(global_seed=1, epoch=0, sample_id="s1", occurrence=1)
    assert first != second


def test_hybrid_seed_is_deterministic() -> None:
    first = derive_hybrid_seed(global_seed=1, sample_id="s1", variant=0, epoch=0, occurrence=0, online_probability=0.3)
    second = derive_hybrid_seed(global_seed=1, sample_id="s1", variant=0, epoch=0, occurrence=0, online_probability=0.3)
    assert first == second


def test_hybrid_seed_zero_probability_always_matches_offline() -> None:
    for epoch in range(5):
        hybrid = derive_hybrid_seed(
            global_seed=1, sample_id="s1", variant=2, epoch=epoch, occurrence=0, online_probability=0.0
        )
        offline = derive_offline_seed(global_seed=1, sample_id="s1", variant=2)
        assert hybrid == offline


def test_hybrid_seed_probability_one_always_matches_online() -> None:
    for epoch in range(5):
        hybrid = derive_hybrid_seed(
            global_seed=1, sample_id="s1", variant=2, epoch=epoch, occurrence=0, online_probability=1.0
        )
        online = derive_online_seed(global_seed=1, epoch=epoch, sample_id="s1", occurrence=0)
        assert hybrid == online


def test_hybrid_seed_branch_choice_varies_across_epochs() -> None:
    # With a mid-range probability, some epochs must land on each branch across a wide sweep.
    offline_hits = 0
    online_hits = 0
    for epoch in range(50):
        hybrid = derive_hybrid_seed(
            global_seed=1, sample_id="s1", variant=0, epoch=epoch, occurrence=0, online_probability=0.5
        )
        offline = derive_offline_seed(global_seed=1, sample_id="s1", variant=0)
        online = derive_online_seed(global_seed=1, epoch=epoch, sample_id="s1", occurrence=0)
        if hybrid == offline:
            offline_hits += 1
        elif hybrid == online:
            online_hits += 1
    assert offline_hits > 0
    assert online_hits > 0


def test_hybrid_seed_invalid_probability_raises() -> None:
    with pytest.raises(ValueError, match=r"online_probability must be in \[0, 1\]"):
        derive_hybrid_seed(global_seed=1, sample_id="s1", variant=0, epoch=0, occurrence=0, online_probability=1.5)
