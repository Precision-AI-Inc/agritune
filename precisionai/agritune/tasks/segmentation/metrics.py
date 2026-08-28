# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Segmentation metrics: mean IoU, per-class IoU, Dice/F1, pixel accuracy, confusion matrix.

:class:`SegmentationMetric` accumulates a confusion matrix across batches and derives every
statistic from it at :meth:`~SegmentationMetric.compute` time — satisfies
:class:`~precisionai.agritune.schemas.protocols.Metric`.
"""

import torch


class SegmentationMetric:
    """Accumulates a confusion matrix and computes mIoU, per-class IoU, Dice/F1, pixel accuracy.

    Parameters
    ----------
    num_classes : int
        Number of segmentation classes.
    ignore_index : int | None, optional
        Pixels with this target value are excluded from every statistic.
    """

    def __init__(self, *, num_classes: int, ignore_index: int | None = None) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self._confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._confusion.zero_()

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate one batch.

        Parameters
        ----------
        outputs : torch.Tensor
            Either per-class logits, shape ``(B, num_classes, H, W)`` (argmax'd internally), or
            already-computed predictions, shape ``(B, H, W)``.
        targets : torch.Tensor
            Ground-truth class indices, shape ``(B, H, W)``.
        """
        predictions = outputs.argmax(dim=1) if outputs.ndim == 4 else outputs
        valid = (
            targets != self.ignore_index
            if self.ignore_index is not None
            else torch.ones_like(targets, dtype=torch.bool)
        )

        predictions = predictions[valid].reshape(-1)
        targets = targets[valid].reshape(-1)
        indices = targets * self.num_classes + predictions
        counts = torch.bincount(indices, minlength=self.num_classes**2)
        self._confusion += counts.reshape(self.num_classes, self.num_classes).to(self._confusion.dtype)

    def compute(self) -> dict[str, float]:
        """Return every statistic derived from the accumulated confusion matrix so far.

        Returns
        -------
        dict[str, float]
            ``mean_iou``, ``mean_dice``, ``pixel_accuracy``, plus ``iou_class_{i}`` and
            ``dice_class_{i}`` for every class ``i``. A class that never appears in either
            predictions or targets reports an IoU/Dice of ``0.0`` rather than ``NaN``.
        """
        confusion = self._confusion.float()
        true_positive = confusion.diag()
        predicted_totals = confusion.sum(dim=0)
        actual_totals = confusion.sum(dim=1)
        union = predicted_totals + actual_totals - true_positive

        iou_per_class = true_positive / union.clamp(min=1)
        dice_per_class = 2 * true_positive / (predicted_totals + actual_totals).clamp(min=1)
        pixel_accuracy = true_positive.sum() / confusion.sum().clamp(min=1)

        result = {
            "mean_iou": iou_per_class.mean().item(),
            "mean_dice": dice_per_class.mean().item(),
            "pixel_accuracy": pixel_accuracy.item(),
        }
        for class_index in range(self.num_classes):
            result[f"iou_class_{class_index}"] = iou_per_class[class_index].item()
            result[f"dice_class_{class_index}"] = dice_per_class[class_index].item()
        return result

    def confusion_matrix(self) -> torch.Tensor:
        """Return a copy of the accumulated confusion matrix, rows=targets, cols=predictions."""
        return self._confusion.clone()
