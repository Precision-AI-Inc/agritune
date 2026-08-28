# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Deterministic train/validation/test split strategies.

``random_split`` assigns each sample independently; ``grouped_split`` keeps every sample sharing
a metadata key (e.g. ``field_id``) in the same split. Grouped splits matter for agricultural data:
nearby frames/images of the same field can be nearly identical, and a random split would leak
near-duplicates between train and eval. See ``docs/datasets.md``.
"""

import hashlib
from dataclasses import dataclass

from precisionai.agritune.data.manifest import ManifestRow


@dataclass
class SplitAssignment:
    """The result of a split strategy: sample IDs assigned to each subset.

    Attributes
    ----------
    train : list[str]
    val : list[str]
    test : list[str]
    """

    train: list[str]
    val: list[str]
    test: list[str]


def _stable_hash_fraction(key: str, seed: int) -> float:
    """Deterministically map a string key to a float in ``[0, 1)``, stable across processes."""
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _validate_fractions(train_fraction: float, val_fraction: float) -> None:
    if not (0 < train_fraction < 1):
        raise ValueError(f"train_fraction must be in (0, 1); got {train_fraction}")
    if not (0 <= val_fraction < 1):
        raise ValueError(f"val_fraction must be in [0, 1); got {val_fraction}")
    if train_fraction + val_fraction >= 1:
        raise ValueError(
            f"train_fraction + val_fraction must be < 1 (need room for test); got {train_fraction + val_fraction}"
        )


def random_split(
    rows: list[ManifestRow],
    *,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> SplitAssignment:
    """Assign each sample to train/val/test independently, by a seeded hash of its sample ID.

    Parameters
    ----------
    rows : list[ManifestRow]
        Manifest rows to split.
    train_fraction : float, optional
        Target fraction of samples assigned to train.
    val_fraction : float, optional
        Target fraction of samples assigned to val. The remainder goes to test.
    seed : int, optional
        Seed for the deterministic hash; the same rows/fractions/seed always produce the same
        split.

    Returns
    -------
    SplitAssignment
        Sample IDs assigned to each subset.
    """
    _validate_fractions(train_fraction, val_fraction)
    train, val, test = [], [], []
    for row in rows:
        fraction = _stable_hash_fraction(row.sample_id, seed)
        if fraction < train_fraction:
            train.append(row.sample_id)
        elif fraction < train_fraction + val_fraction:
            val.append(row.sample_id)
        else:
            test.append(row.sample_id)
    return SplitAssignment(train=train, val=val, test=test)


def grouped_split(
    rows: list[ManifestRow],
    *,
    group_by: str,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> SplitAssignment:
    """Assign whole groups (e.g. all samples sharing a ``field_id``) to the same split.

    Parameters
    ----------
    rows : list[ManifestRow]
        Manifest rows to split.
    group_by : str
        Metadata key identifying the group (e.g. ``"field_id"``). Every row must carry this key.
    train_fraction : float, optional
        Target fraction of *groups* assigned to train.
    val_fraction : float, optional
        Target fraction of *groups* assigned to val. The remainder goes to test.
    seed : int, optional
        Seed for the deterministic hash.

    Returns
    -------
    SplitAssignment
        Sample IDs assigned to each subset; every sample sharing a ``group_by`` value is in the
        same subset.

    Raises
    ------
    ValueError
        If any row is missing the ``group_by`` metadata key.
    """
    _validate_fractions(train_fraction, val_fraction)

    groups: dict[str, list[str]] = {}
    for row in rows:
        group_value = row.metadata.get(group_by)
        if not group_value:
            raise ValueError(f"sample '{row.sample_id}' is missing grouping metadata '{group_by}'")
        groups.setdefault(group_value, []).append(row.sample_id)

    train, val, test = [], [], []
    for group_value, sample_ids in groups.items():
        fraction = _stable_hash_fraction(group_value, seed)
        if fraction < train_fraction:
            train.extend(sample_ids)
        elif fraction < train_fraction + val_fraction:
            val.extend(sample_ids)
        else:
            test.extend(sample_ids)
    return SplitAssignment(train=train, val=val, test=test)


def detect_group_leakage(rows: list[ManifestRow], split: SplitAssignment, *, group_by: str) -> list[str]:
    """Return group values whose samples are not all assigned to the same split.

    Useful even when ``split`` came from :func:`random_split`: it catches near-duplicate
    agricultural frames (same field, same day) leaking across train/val/test.

    Parameters
    ----------
    rows : list[ManifestRow]
        The manifest rows the split was computed over.
    split : SplitAssignment
        A resolved split assignment.
    group_by : str
        Metadata key identifying the group to check (e.g. ``"field_id"``).

    Returns
    -------
    list[str]
        Group values that appear in more than one of train/val/test, sorted.
    """
    sample_to_split: dict[str, str] = {}
    for sample_id in split.train:
        sample_to_split[sample_id] = "train"
    for sample_id in split.val:
        sample_to_split[sample_id] = "val"
    for sample_id in split.test:
        sample_to_split[sample_id] = "test"

    group_to_splits: dict[str, set[str]] = {}
    for row in rows:
        group_value = row.metadata.get(group_by)
        split_name = sample_to_split.get(row.sample_id)
        if not group_value or split_name is None:
            continue
        group_to_splits.setdefault(group_value, set()).add(split_name)

    return sorted(group for group, splits in group_to_splits.items() if len(splits) > 1)
