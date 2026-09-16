# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Segmentation losses: CrossEntropy, BCEWithLogits, Dice, and CE/BCE + Dice combinations.

Every loss takes ``logits`` of shape ``(B, num_classes, H, W)`` and ``targets`` of shape
``(B, H, W)`` holding integer class indices (a single-label per-pixel class map) — including the
BCE variant, which internally one-hot encodes ``targets`` and applies a per-class independent
binary loss. ``ignore_index`` excludes pixels (e.g. a "void"/unlabeled class) from every loss.

A decoder's ``output_size`` (its fixed upsample target, set once at construction) does not
necessarily match every batch's targets — most notably, validation targets are never resized by
geometric augmentation while training targets may be (``resize``/``random_crop``), so the two
splits can legitimately disagree on spatial size. ``SegmentationLoss.forward`` bilinearly resizes
``logits`` to ``targets``'s spatial size whenever they differ, rather than requiring an exact match.
"""

from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import nn
from torch.nn import functional

LossName = Literal["ce", "bce", "dice", "ce_dice", "bce_dice"]


class DiceLoss(nn.Module):
    """Soft Dice loss over one-hot-encoded, per-pixel class targets.

    Parameters
    ----------
    num_classes : int
        Number of segmentation classes.
    ignore_index : int | None, optional
        Pixels with this target value are excluded from the loss.
    smooth : float, optional
        Additive smoothing term avoiding division by zero for classes absent from a batch.
    """

    def __init__(self, *, num_classes: int, ignore_index: int | None = None, smooth: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return the mean per-class soft Dice loss (``1 - Dice coefficient``)."""
        probs = functional.softmax(logits, dim=1)
        one_hot, valid_mask = _one_hot_targets(targets, num_classes=self.num_classes, ignore_index=self.ignore_index)

        probs = probs * valid_mask
        one_hot = one_hot * valid_mask

        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        cardinality = probs.sum(dims) + one_hot.sum(dims)
        dice_per_class = (2 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_per_class.mean()


def _validate_targets(targets: torch.Tensor, *, num_classes: int, valid: torch.Tensor) -> None:
    invalid = valid & ((targets < 0) | (targets >= num_classes))
    if torch.any(invalid):
        labels = sorted(int(label) for label in torch.unique(targets[invalid]).tolist())
        raise ValueError(f"targets contain labels outside [0, {num_classes}): {labels}")


def _one_hot_targets(
    targets: torch.Tensor, *, num_classes: int, ignore_index: int | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(one_hot, valid_mask)``, both shape ``(B, num_classes, H, W)``."""
    valid = targets != ignore_index if ignore_index is not None else torch.ones_like(targets, dtype=torch.bool)
    _validate_targets(targets, num_classes=num_classes, valid=valid)
    clamped = targets.clamp(min=0, max=num_classes - 1)
    one_hot = functional.one_hot(clamped, num_classes=num_classes).permute(0, 3, 1, 2).float()
    return one_hot, valid.unsqueeze(1).float()


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = -100,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the standard per-pixel cross-entropy loss."""
    valid = targets != ignore_index
    _validate_targets(targets, num_classes=logits.shape[1], valid=valid)
    if not torch.any(valid):
        return logits.sum() * 0.0
    return functional.cross_entropy(logits, targets, weight=class_weights, ignore_index=ignore_index)


def bce_with_logits_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: int | None = None,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-class independent binary cross-entropy against one-hot-encoded targets."""
    one_hot, valid_mask = _one_hot_targets(targets, num_classes=num_classes, ignore_index=ignore_index)
    pos_weight = class_weights.view(-1, 1, 1) if class_weights is not None else None
    per_element = functional.binary_cross_entropy_with_logits(logits, one_hot, pos_weight=pos_weight, reduction="none")
    per_element = per_element * valid_mask
    return per_element.sum() / (valid_mask.sum() * num_classes).clamp(min=1)


@dataclass
class SegmentationLossConfig:
    """Configuration for :class:`SegmentationLoss`.

    Attributes
    ----------
    name : LossName
        Which loss (or combination) to compute.
    ignore_index : int
        Pixel value to exclude from cross-entropy/BCE/Dice terms.
    class_weights : torch.Tensor | None
        Per-class weights — ``weight`` for cross-entropy, ``pos_weight`` for BCE.
    ce_weight : float
        Weight applied to the CE/BCE term in a combined loss.
    dice_weight : float
        Weight applied to the Dice term in a combined loss.
    """

    name: LossName = "ce"
    ignore_index: int = -100
    class_weights: torch.Tensor | None = None
    ce_weight: float = 1.0
    dice_weight: float = 1.0


class SegmentationLoss(nn.Module):
    """Dispatches to the loss (or CE/BCE + Dice combination) named in ``config``.

    Parameters
    ----------
    config : SegmentationLossConfig
        Loss selection and weighting.
    num_classes : int
        Number of segmentation classes.
    """

    def __init__(self, config: SegmentationLossConfig, *, num_classes: int) -> None:
        super().__init__()
        self._config = config
        self._num_classes = num_classes
        self._dice = DiceLoss(num_classes=num_classes, ignore_index=config.ignore_index)
        # Registered as a buffer (not just kept on `config`) so `Module.to(device)` carries it along
        # automatically — otherwise a GPU-resident model would call `cross_entropy`/`bce_with_logits`
        # with CPU weights against CUDA logits and fail with a device-mismatch error.
        self.register_buffer("_class_weights", config.class_weights, persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the configured loss for one batch."""
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = functional.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
        config = self._config
        # nn.Module's __getattr__ is typed as returning `Tensor | Module` for any registered
        # buffer/submodule name, since it can't know which this one is; this buffer is always a
        # plain tensor or None (see __init__), never a submodule.
        class_weights = cast("torch.Tensor | None", self._class_weights)
        if config.name == "ce":
            return cross_entropy_loss(logits, targets, ignore_index=config.ignore_index, class_weights=class_weights)
        if config.name == "bce":
            return bce_with_logits_loss(
                logits,
                targets,
                num_classes=self._num_classes,
                ignore_index=config.ignore_index,
                class_weights=class_weights,
            )
        if config.name == "dice":
            return self._dice(logits, targets)
        if config.name == "ce_dice":
            ce = cross_entropy_loss(logits, targets, ignore_index=config.ignore_index, class_weights=class_weights)
            return config.ce_weight * ce + config.dice_weight * self._dice(logits, targets)
        if config.name == "bce_dice":
            bce = bce_with_logits_loss(
                logits,
                targets,
                num_classes=self._num_classes,
                ignore_index=config.ignore_index,
                class_weights=class_weights,
            )
            return config.ce_weight * bce + config.dice_weight * self._dice(logits, targets)
        raise ValueError(f"unsupported loss name: {config.name!r}")  # pragma: no cover — exhaustive over LossName
