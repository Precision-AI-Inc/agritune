# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""``TrainingBatch`` — the collated unit the trainer, evaluator, and prefetcher all consume.

Shared by :class:`~precisionai.agritune.training.trainer.Trainer`,
:func:`~precisionai.agritune.training.evaluator.evaluate`, and
:class:`~precisionai.agritune.training.prefetch.PrefetchingFeatureLoader`. Kept in its own module
(rather than alongside ``evaluate``) purely to break an import cycle: ``prefetch`` needs the type
and ``evaluator`` needs ``prefetch``.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from precisionai.agritune.schemas.samples import PreparedSample


@dataclass
class TrainingBatch:
    """One batch as the trainer/evaluator loop consumes it.

    Attributes
    ----------
    samples : Sequence[PreparedSample]
        Augmented samples to fetch features for.
    targets : torch.Tensor
        Ground-truth targets, stacked to match ``samples`` order (e.g. ``(B, H, W)`` class maps
        for segmentation).
    """

    samples: Sequence[PreparedSample]
    targets: torch.Tensor

    def pin_memory(self) -> "TrainingBatch":
        """Return a copy with ``targets`` in pinned memory, for a ``DataLoader(pin_memory=True)``.

        ``DataLoader`` only knows how to pin plain tensors/mappings/sequences out of the box; a
        custom collated type like this one must implement this method itself, or ``pin_memory``
        silently pins nothing. ``samples`` holds PIL images, not tensors, so there is nothing in it
        to pin.
        """
        return TrainingBatch(samples=self.samples, targets=self.targets.pin_memory())
