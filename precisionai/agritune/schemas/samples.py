# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Dataset-facing sample representations.

A :class:`Sample` is what a dataset adapter (``precisionai.agritune.data``) yields. A
:class:`PreparedSample` is what an augmentation pipeline (``precisionai.agritune.augmentations``)
produces from a ``Sample`` — same shape of information, plus a record of what was done to it.

Agricultural metadata (``farm_id``, ``field_id``, ``capture_date``, ``sensor``, ``crop``,
``growth_stage``, ``location``, ...) lives in the free-form ``metadata`` dict. None of it is
mandatory — a dataset adapter that has none of these fields is still a valid ``Sample``.
"""

from dataclasses import dataclass, field
from typing import Any

from precisionai.agritune.schemas.augmentation import AugmentationRecord


@dataclass
class Sample:
    """A single dataset item before augmentation.

    Attributes
    ----------
    sample_id : str
        Stable, unique identifier for this sample within its dataset. Used to derive
        deterministic augmentation seeds and feature cache keys.
    image : Any
        The input image, in whatever representation the dataset adapter produces (e.g. a
        ``PIL.Image.Image``, ``numpy.ndarray``, or ``torch.Tensor``) — narrowed by the concrete
        dataset adapter, not by this schema.
    target : Any
        The training target (e.g. a segmentation mask), in whatever representation the dataset
        adapter produces.
    metadata : dict[str, Any]
        Free-form sample metadata. May include agricultural fields such as ``farm_id``,
        ``field_id``, ``capture_date``, ``sensor``, ``crop``, ``growth_stage``, or ``location``,
        but none are required.
    """

    sample_id: str
    image: Any
    target: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedSample:
    """A :class:`Sample` after augmentation, ready for feature extraction.

    Attributes
    ----------
    sample_id : str
        Carried over unchanged from the source :class:`Sample`.
    image : Any
        The augmented image.
    target : Any
        The augmented target, geometrically aligned with ``image`` (masks receive only geometric
        transforms — see ``docs/augmentation.md``).
    augmentation_metadata : AugmentationRecord | None
        Deterministic record of what augmentation was applied (seed, transform list) — sufficient
        to reproduce this exact ``image``/``target`` pair offline. ``None`` when no augmentation
        was applied (``augmentation.mode: none``).
    """

    sample_id: str
    image: Any
    target: Any
    augmentation_metadata: AugmentationRecord | None = None
