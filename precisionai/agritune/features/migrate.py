# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Feature store migration — powers ``agritune features migrate``.

Copies every entry from one feature store into another (typically ``DirectoryFeatureStore`` ->
``ShardedFeatureStore``, moving a store past development scale) as pure local I/O: every entry
read from the source is already a fully computed ``EncoderFeatures``, so migrating never calls the
encoder. Resumable — entries the destination already has are skipped, so an interrupted migration
can simply be re-run.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from precisionai.agritune.schemas.features import EncoderFeatures


class _ReadableStore(Protocol):
    def list_keys(self) -> list[str]: ...
    def read(self, key: str) -> EncoderFeatures: ...


class _WritableStore(Protocol):
    def has(self, key: str) -> bool: ...
    def write(self, key: str, features: EncoderFeatures) -> None: ...
    def flush(self) -> None: ...


@dataclass
class MigrationReport:
    """The result of migrating every entry from one feature store into another.

    Attributes
    ----------
    total : int
        Number of keys found in the source store.
    migrated : int
        Number of entries actually copied.
    skipped : int
        Number of entries the destination already had (not re-copied).
    """

    total: int
    migrated: int = 0
    skipped: int = 0


def migrate_store(
    source: _ReadableStore,
    dest: _WritableStore,
    *,
    on_progress: Callable[[MigrationReport], None] | None = None,
) -> MigrationReport:
    """Copy every entry from ``source`` into ``dest``, skipping keys ``dest`` already has.

    Parameters
    ----------
    source : DirectoryFeatureStore | ShardedFeatureStore
        Store to read every entry from; left untouched.
    dest : DirectoryFeatureStore | ShardedFeatureStore
        Store to write into — may already be partially populated from an interrupted prior run of
        this same migration, in which case those keys are skipped rather than re-copied.
    on_progress : Callable[[MigrationReport], None] | None, optional
        Called after every key with running totals.

    Returns
    -------
    MigrationReport
    """
    keys = source.list_keys()
    report = MigrationReport(total=len(keys))
    for key in keys:
        if dest.has(key):
            report.skipped += 1
        else:
            dest.write(key, source.read(key))
            report.migrated += 1
        if on_progress is not None:
            on_progress(report)
    dest.flush()
    return report
