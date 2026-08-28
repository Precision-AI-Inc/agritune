# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.augmentations.image.seeding."""

from precisionai.agritune.augmentations.image.seeding import derive_offline_seed, derive_online_seed


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
