# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Evaluation loop — runs a :class:`Task` over batches with no gradient tracking.

Shared by :class:`~precisionai.agritune.training.trainer.Trainer`'s validation pass and by
``agritune evaluate``.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch

from precisionai.agritune.logging import get_logger, progress_iter
from precisionai.agritune.schemas.protocols import FeatureProvider, Metric, Task
from precisionai.agritune.schemas.samples import PreparedSample

logger = get_logger(__name__)


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


def evaluate(
    task: Task,
    feature_provider: FeatureProvider,
    batches: Iterable[TrainingBatch],
    metric: Metric,
    *,
    show_progress: bool = False,
) -> dict[str, float]:
    """Run ``task`` over every batch with gradients disabled, accumulating ``metric`` and loss.

    Parameters
    ----------
    task : Task
        The task to evaluate (its decoder is used in eval-appropriate mode by the caller, e.g.
        ``decoder.eval()``, before calling this — this function does not toggle module mode
        itself since ``Task`` is a protocol, not necessarily an ``nn.Module``).
    feature_provider : FeatureProvider
        Supplies features for each batch's samples.
    batches : Iterable[TrainingBatch]
        Batches to evaluate over.
    metric : Metric
        Accumulates statistics across every batch; not reset here — callers own its lifecycle.
    show_progress : bool, optional
        Render a ``tqdm`` bar over ``batches``. Defaults to ``False`` so headless callers (e.g.
        the API) see no terminal output; the caller decides (e.g. gating on
        ``DistributedContext.is_main_process``) when this runs under multiple processes.

    Returns
    -------
    dict[str, float]
        ``metric.compute()`` after every batch has been processed, plus a ``"loss"`` entry: the
        mean of ``task.compute_loss(outputs, targets)`` over every batch, weighted by batch size
        (so a smaller trailing batch doesn't skew the average). ``0.0`` if ``batches`` is empty.
    """
    loss_total = 0.0
    sample_count = 0
    with torch.no_grad():
        for batch in progress_iter(batches, desc="evaluate", unit="batch", disable=not show_progress):
            features = feature_provider.get_features(batch.samples)
            outputs = task.forward(features)
            metric.update(outputs, batch.targets)
            batch_size = len(batch.samples)
            loss_total += task.compute_loss(outputs, batch.targets).item() * batch_size
            sample_count += batch_size

    result = metric.compute()
    result["loss"] = loss_total / sample_count if sample_count > 0 else 0.0
    logger.info("evaluated %d sample(s): %s", sample_count, result)
    return result
