# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Evaluation loop — runs a :class:`Task` over batches with no gradient tracking.

Shared by :class:`~precisionai.agritune.training.trainer.Trainer`'s validation pass and by
``agritune evaluate``.
"""

from collections.abc import Iterable

import torch

from precisionai.agritune.logging import get_logger, progress_iter
from precisionai.agritune.schemas.protocols import FeatureProvider, Metric, Task
from precisionai.agritune.training.batch import TrainingBatch
from precisionai.agritune.training.prefetch import PrefetchingFeatureLoader

logger = get_logger(__name__)

__all__ = ["TrainingBatch", "evaluate"]


def evaluate(
    task: Task,
    feature_provider: FeatureProvider,
    batches: Iterable[TrainingBatch],
    metric: Metric,
    *,
    device: torch.device | None = None,
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
    device : torch.device | None, optional
        Moves each batch's features and targets here before ``task.forward``/``compute_loss`` —
        must match wherever ``task``'s own parameters live. ``None`` (the default) leaves them
        wherever ``feature_provider`` and ``batches`` already put them (e.g. both on CPU).
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
    resolved_device = device if device is not None else torch.device("cpu")
    non_blocking = device is not None and device.type == "cuda"
    prefetcher = PrefetchingFeatureLoader(batches, feature_provider, device=resolved_device, non_blocking=non_blocking)
    progress = progress_iter(prefetcher, desc="evaluate", unit="batch", disable=not show_progress)
    with torch.no_grad():
        for batch, features, targets in progress:
            outputs = task.forward(features)
            metric.update(outputs, targets)
            batch_size = len(batch.samples)
            loss_total += task.compute_loss(outputs, targets).item() * batch_size
            sample_count += batch_size

    result = metric.compute()
    result["loss"] = loss_total / sample_count if sample_count > 0 else 0.0
    logger.info("evaluated %d sample(s): %s", sample_count, result)
    return result
