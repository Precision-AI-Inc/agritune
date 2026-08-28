# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Deterministic seed derivation for image augmentation.

Offline augmentation derives a seed from ``global_seed + sample_id + variant`` so the same variant
of the same sample always augments identically. Online augmentation additionally mixes in the
epoch and an occurrence counter, so repeated epochs (and repeated views of the same sample within
an epoch, e.g. oversampling) still produce distinct-but-reproducible augmentations. Hybrid mode
picks deterministically, per (sample, epoch, occurrence), between reusing an offline variant and
deriving a fresh online seed — see :func:`derive_hybrid_seed`.
"""

import hashlib


def _derive_seed(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _stable_fraction(*parts: object) -> float:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


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


def derive_hybrid_seed(
    *, global_seed: int, sample_id: str, variant: int, epoch: int, occurrence: int, online_probability: float
) -> int:
    """Derive a seed for hybrid mode: mostly a reused offline variant, sometimes fresh online.

    A hash of ``(global_seed, sample_id, epoch, occurrence)`` — independent of the seed each
    branch itself derives — deterministically decides which branch applies, so the same
    ``(sample, epoch, occurrence)`` always makes the same choice: this lets most epochs reuse one
    of a fixed pool of ``variant`` offline variants (compatible with an offline feature cache),
    while a configurable fraction of epochs still inject fresh online diversity.

    Parameters
    ----------
    global_seed : int
        The run-level seed.
    sample_id : str
        The sample's stable identifier.
    variant : int
        Which offline variant to reuse when the offline branch is chosen.
    epoch : int
        The current training epoch.
    occurrence : int
        Which occurrence of this sample within the epoch this is.
    online_probability : float
        Probability in ``[0, 1]`` that a given ``(sample, epoch, occurrence)`` takes the fresh
        online branch instead of reusing the offline variant.

    Returns
    -------
    int
        A deterministic, uniformly distributed seed.

    Raises
    ------
    ValueError
        If ``online_probability`` is outside ``[0, 1]``.
    """
    if not 0.0 <= online_probability <= 1.0:
        raise ValueError(f"online_probability must be in [0, 1]; got {online_probability}")

    fraction = _stable_fraction("hybrid-branch", global_seed, sample_id, epoch, occurrence)
    if fraction < online_probability:
        return derive_online_seed(global_seed=global_seed, epoch=epoch, sample_id=sample_id, occurrence=occurrence)
    return derive_offline_seed(global_seed=global_seed, sample_id=sample_id, variant=variant)
