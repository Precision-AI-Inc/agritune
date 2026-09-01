# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Evaluation loop — runs a :class:`Task` over batches with no gradient tracking.

Shared by :class:`~precisionai.agritune.training.trainer.Trainer`'s validation pass and by
``agritune evaluate``.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch

from precisionai.agritune.schemas.protocols import FeatureProvider, Metric, Task
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


def evaluate(
    task: Task,
    feature_provider: FeatureProvider,
    batches: Iterable[TrainingBatch],
    metric: Metric,
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
        for batch in batches:
            features = feature_provider.get_features(batch.samples)
            outputs = task.forward(features)
            metric.update(outputs, batch.targets)
            batch_size = len(batch.samples)
            loss_total += task.compute_loss(outputs, batch.targets).item() * batch_size
            sample_count += batch_size

    result = metric.compute()
    result["loss"] = loss_total / sample_count if sample_count > 0 else 0.0
    return result
