# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Segmentation metrics: mean IoU, per-class IoU/precision/recall, Dice/F1, pixel accuracy, confusion matrix.

:class:`SegmentationMetric` accumulates a confusion matrix across batches and derives every
statistic from it at :meth:`~SegmentationMetric.compute` time — satisfies
:class:`~precisionai.agritune.schemas.protocols.Metric`.
"""

import torch
from torch.nn import functional


def _out_of_range_labels(values: torch.Tensor, *, num_classes: int) -> list[int]:
    invalid = (values < 0) | (values >= num_classes)
    return sorted(int(label) for label in torch.unique(values[invalid]).tolist()) if torch.any(invalid) else []


class SegmentationMetric:
    """Accumulates a confusion matrix and computes mIoU, per-class IoU/precision/recall, Dice/F1, pixel accuracy.

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
            already-computed predictions, shape ``(B, H, W)``. Logits are bilinearly resized to
            ``targets``'s spatial size first if the two disagree — see ``losses.py``'s module
            docstring on why a decoder's fixed ``output_size`` doesn't always match every batch's
            targets (e.g. validation vs. geometrically-augmented training targets).
        targets : torch.Tensor
            Ground-truth class indices, shape ``(B, H, W)``.
        """
        if outputs.ndim not in (3, 4):
            raise ValueError(f"outputs must have shape (B, H, W) or (B, C, H, W); got {tuple(outputs.shape)}")
        if outputs.ndim == 4 and outputs.shape[1] != self.num_classes:
            raise ValueError(f"logits class dimension must be {self.num_classes}; got {outputs.shape[1]}")
        if outputs.ndim == 4 and outputs.shape[-2:] != targets.shape[-2:]:
            outputs = functional.interpolate(outputs, size=targets.shape[-2:], mode="bilinear", align_corners=False)
        predictions = outputs.argmax(dim=1) if outputs.ndim == 4 else outputs
        if predictions.shape != targets.shape:
            raise ValueError(
                f"predictions and targets must share shape (B, H, W); got {tuple(predictions.shape)} and "
                f"{tuple(targets.shape)}"
            )
        valid = (
            targets != self.ignore_index
            if self.ignore_index is not None
            else torch.ones_like(targets, dtype=torch.bool)
        )

        predictions = predictions[valid].reshape(-1)
        targets = targets[valid].reshape(-1)
        invalid_targets = _out_of_range_labels(targets, num_classes=self.num_classes)
        if invalid_targets:
            raise ValueError(f"targets contain labels outside [0, {self.num_classes}): {invalid_targets}")
        invalid_predictions = _out_of_range_labels(predictions, num_classes=self.num_classes)
        if invalid_predictions:
            raise ValueError(f"predictions contain labels outside [0, {self.num_classes}): {invalid_predictions}")
        indices = targets * self.num_classes + predictions
        counts = torch.bincount(indices, minlength=self.num_classes**2)
        # Follows wherever outputs/targets actually are (e.g. a CUDA device once Trainer starts
        # moving batches there) rather than forcing every update() call to transfer back to
        # whatever device the metric happened to be constructed on.
        if self._confusion.device != counts.device:
            self._confusion = self._confusion.to(counts.device)
        self._confusion += counts.reshape(self.num_classes, self.num_classes).to(self._confusion.dtype)

    def compute(self) -> dict[str, float]:
        """Return every statistic derived from the accumulated confusion matrix so far.

        Returns
        -------
        dict[str, float]
            ``mean_iou``, ``mean_dice``, ``mean_precision``, ``mean_recall``, ``pixel_accuracy``,
            plus ``iou_class_{i}``, ``dice_class_{i}``, ``precision_class_{i}``, and
            ``recall_class_{i}`` for every class ``i``. A class that never appears in either
            predictions or targets reports ``0.0`` rather than ``NaN`` for every per-class stat.
        """
        confusion = self._confusion.float()
        true_positive = confusion.diag()
        predicted_totals = confusion.sum(dim=0)
        actual_totals = confusion.sum(dim=1)
        union = predicted_totals + actual_totals - true_positive

        iou_per_class = true_positive / union.clamp(min=1)
        dice_per_class = 2 * true_positive / (predicted_totals + actual_totals).clamp(min=1)
        precision_per_class = true_positive / predicted_totals.clamp(min=1)
        recall_per_class = true_positive / actual_totals.clamp(min=1)
        pixel_accuracy = true_positive.sum() / confusion.sum().clamp(min=1)

        result = {
            "mean_iou": iou_per_class.mean().item(),
            "mean_dice": dice_per_class.mean().item(),
            "mean_precision": precision_per_class.mean().item(),
            "mean_recall": recall_per_class.mean().item(),
            "pixel_accuracy": pixel_accuracy.item(),
        }
        for class_index in range(self.num_classes):
            result[f"iou_class_{class_index}"] = iou_per_class[class_index].item()
            result[f"dice_class_{class_index}"] = dice_per_class[class_index].item()
            result[f"precision_class_{class_index}"] = precision_per_class[class_index].item()
            result[f"recall_class_{class_index}"] = recall_per_class[class_index].item()
        return result

    def confusion_matrix(self) -> torch.Tensor:
        """Return a copy of the accumulated confusion matrix, rows=targets, cols=predictions."""
        return self._confusion.clone()
