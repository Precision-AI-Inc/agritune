# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Feature store integrity verification — powers ``agritune features verify``.

Detects a corrupted feature shard (bit rot, a truncated write, manual tampering) by recomputing
each entry's checksum and comparing it to what was recorded at write time.
"""

from dataclasses import dataclass, field
from typing import Protocol


class _VerifiableStore(Protocol):
    def list_keys(self) -> list[str]: ...
    def verify(self, key: str) -> bool: ...


@dataclass
class VerificationReport:
    """The result of verifying every entry in a feature store.

    Attributes
    ----------
    total : int
        Number of entries checked.
    corrupted_keys : list[str]
        Keys whose stored checksum did not match their actual file contents.
    """

    total: int
    corrupted_keys: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Return ``True`` if no entry was found corrupted."""
        return not self.corrupted_keys


def verify_store(store: _VerifiableStore) -> VerificationReport:
    """Verify every entry in ``store`` and report which (if any) are corrupted.

    Parameters
    ----------
    store : DirectoryFeatureStore | ShardedFeatureStore
        The store to verify.

    Returns
    -------
    VerificationReport
    """
    keys = store.list_keys()
    corrupted = [key for key in keys if not store.verify(key)]
    return VerificationReport(total=len(keys), corrupted_keys=corrupted)
