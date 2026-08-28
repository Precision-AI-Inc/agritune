# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Deterministic augmentation record: what was applied to a sample, and with what seed.

Kept in ``schemas`` (rather than ``augmentations``) because :class:`~precisionai.agritune.schemas.samples.PreparedSample`
references it and must not depend on the augmentation package's implementation.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TransformRecord:
    """One applied transform, recorded for exact offline reproduction.

    Attributes
    ----------
    name : str
        Transform identifier (e.g. ``"random_crop"``, ``"horizontal_flip"``, ``"brightness"``).
    params : dict[str, Any]
        The concrete parameters sampled for this application (e.g. crop box, flip flag, jitter
        factor) — enough to replay the transform deterministically.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class AugmentationRecord:
    """A deterministic, replayable record of the augmentation applied to one sample.

    Attributes
    ----------
    seed : int
        The exact seed used to derive every random choice in ``transforms``. See
        ``docs/augmentation.md`` for the seed derivation formula (offline vs. online).
    transforms : list[TransformRecord]
        The transforms applied, in application order. Empty when ``augmentation.mode: none``.
    """

    seed: int
    transforms: list[TransformRecord] = field(default_factory=list)
