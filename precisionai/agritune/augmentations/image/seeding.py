# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Deterministic seed derivation for image augmentation.

Offline augmentation derives a seed from ``global_seed + sample_id + variant`` so the same variant
of the same sample always augments identically. Online augmentation additionally mixes in the
epoch and an occurrence counter, so repeated epochs (and repeated views of the same sample within
an epoch, e.g. oversampling) still produce distinct-but-reproducible augmentations.
"""

import hashlib


def _derive_seed(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def derive_offline_seed(*, global_seed: int, sample_id: str, variant: int) -> int:
    """Derive a deterministic seed for one offline augmentation variant of one sample.

    Parameters
    ----------
    global_seed : int
        The run-level seed.
    sample_id : str
        The sample's stable identifier.
    variant : int
        Which variant of this sample this is (``0`` .. ``variants_per_sample - 1``).

    Returns
    -------
    int
        A deterministic, uniformly distributed seed.
    """
    return _derive_seed(global_seed, sample_id, variant)


def derive_online_seed(*, global_seed: int, epoch: int, sample_id: str, occurrence: int) -> int:
    """Derive a deterministic seed for one online augmentation of one sample.

    Parameters
    ----------
    global_seed : int
        The run-level seed.
    epoch : int
        The current training epoch.
    sample_id : str
        The sample's stable identifier.
    occurrence : int
        Which occurrence of this sample within the epoch this is (usually ``0``; greater than
        zero only if a sampler can yield the same sample more than once per epoch).

    Returns
    -------
    int
        A deterministic, uniformly distributed seed.
    """
    return _derive_seed(global_seed, epoch, sample_id, occurrence)
