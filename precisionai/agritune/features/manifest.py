# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""A read-only summary of everything currently durable in a :class:`FeatureStore`.

Deliberately not an independently maintained bookkeeping file: the store's own index (the
directory listing for :class:`~precisionai.agritune.features.store.DirectoryFeatureStore`, the
shard index for :class:`~precisionai.agritune.features.store.ShardedFeatureStore`) is the single
source of truth for what has actually been durably written — anything else would risk drifting out
of sync with reality after a crash mid-build. This module only reads that truth back out, for
``agritune features inspect``/``verify`` and for progress reporting.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from precisionai.agritune.features.store import FeatureSummary


class _SummaryReadableStore(Protocol):
    def list_keys(self) -> list[str]: ...
    def read_summary(self, key: str) -> FeatureSummary: ...


@dataclass
class FeatureManifest:
    """Aggregate statistics over every entry in a feature store.

    Attributes
    ----------
    entries : list[FeatureSummary]
        One summary per stored sample.
    """

    entries: list[FeatureSummary]

    def __len__(self) -> int:
        """Return the number of stored entries."""
        return len(self.entries)

    def encoder_models(self) -> Counter[str]:
        """Return a count of entries per ``encoder_model``.

        More than one distinct model present is a sign the cache mixes incompatible encoder
        versions — see ``docs/encoder.md`` on why the encoder revision alone cannot detect this.
        """
        return Counter(entry.encoder_model for entry in self.entries)

    def patch_dims(self) -> Counter[int]:
        """Return a count of entries per observed patch dimension."""
        return Counter(entry.patch_dim for entry in self.entries)

    @classmethod
    def from_store(cls, store: _SummaryReadableStore) -> "FeatureManifest":
        """Build a manifest by reading every entry's summary out of ``store``.

        Parameters
        ----------
        store : DirectoryFeatureStore | ShardedFeatureStore
            The store to summarize.

        Returns
        -------
        FeatureManifest
        """
        return cls(entries=[store.read_summary(key) for key in store.list_keys()])
