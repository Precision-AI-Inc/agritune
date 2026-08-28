# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.tasks.segmentation.losses."""

import torch

from precisionai.agritune.tasks.segmentation.losses import (
    DiceLoss,
    SegmentationLoss,
    SegmentationLossConfig,
    bce_with_logits_loss,
    cross_entropy_loss,
)

_NUM_CLASSES = 3


def _perfect_logits(targets: torch.Tensor, *, num_classes: int, confidence: float = 10.0) -> torch.Tensor:
    """Build logits that confidently predict exactly ``targets`` (target class: +confidence,
    every other class: -confidence — confident under softmax *and* independent-sigmoid readings)."""
    one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
    return (one_hot * 2 - 1) * confidence


def test_cross_entropy_loss_is_low_for_confident_correct_predictions() -> None:
    targets = torch.randint(0, _NUM_CLASSES, (2, 4, 4))
    logits = _perfect_logits(targets, num_classes=_NUM_CLASSES)
    assert cross_entropy_loss(logits, targets).item() < 0.01


def test_cross_entropy_loss_ignore_index_excludes_pixels() -> None:
    # All targets are 0 except one ignored pixel with a wildly wrong, confident logit there.
    # If ignore_index worked, the loss equals the loss over only the correct, confident pixels (~0).
    targets = torch.zeros(1, 2, 2, dtype=torch.long)
    targets[0, 1, 1] = -100
    logits = _perfect_logits(torch.zeros(1, 2, 2, dtype=torch.long), num_classes=_NUM_CLASSES)
    logits[0, :, 1, 1] = torch.tensor([-10.0, 10.0, -10.0])  # confidently wrong at the ignored pixel

    loss = cross_entropy_loss(logits, targets, ignore_index=-100)
    assert loss.item() < 0.01


def test_cross_entropy_loss_class_weights_changes_value() -> None:
    # A mix of classes is required: if every target were the same class, weighting cancels out
    # of the mean-reduction (sum(loss * w) / sum(w) collapses back to the unweighted mean).
    targets = torch.tensor([[[0, 1], [1, 0]]])
    logits = torch.randn(1, _NUM_CLASSES, 2, 2)
    unweighted = cross_entropy_loss(logits, targets)
    weighted = cross_entropy_loss(logits, targets, class_weights=torch.tensor([5.0, 1.0, 1.0]))
    assert not torch.isclose(unweighted, weighted)


def test_dice_loss_is_low_for_confident_correct_predictions() -> None:
    targets = torch.randint(0, _NUM_CLASSES, (2, 8, 8))
    logits = _perfect_logits(targets, num_classes=_NUM_CLASSES)
    loss = DiceLoss(num_classes=_NUM_CLASSES)(logits, targets)
    assert loss.item() < 0.05


def test_dice_loss_is_high_for_confidently_wrong_predictions() -> None:
    # Two classes only, both actually present: with a 3rd never-seen class, smoothing gives it a
    # perfect Dice score by default and dilutes the mean — see DiceLoss's smoothing behavior.
    num_classes = 2
    targets = torch.zeros(2, 8, 8, dtype=torch.long)
    wrong = torch.full((2, 8, 8), 1, dtype=torch.long)
    logits = _perfect_logits(wrong, num_classes=num_classes)
    loss = DiceLoss(num_classes=num_classes)(logits, targets)
    assert loss.item() > 0.9


def test_dice_loss_respects_ignore_index() -> None:
    targets = torch.full((1, 2, 2), -1, dtype=torch.long)  # entirely ignored
    logits = torch.randn(1, _NUM_CLASSES, 2, 2)
    loss = DiceLoss(num_classes=_NUM_CLASSES, ignore_index=-1)(logits, targets)
    assert torch.isfinite(loss)


def test_bce_with_logits_loss_is_low_for_confident_correct_predictions() -> None:
    targets = torch.randint(0, _NUM_CLASSES, (2, 4, 4))
    logits = _perfect_logits(targets, num_classes=_NUM_CLASSES)
    loss = bce_with_logits_loss(logits, targets, num_classes=_NUM_CLASSES)
    assert loss.item() < 0.05


def test_segmentation_loss_ce() -> None:
    targets = torch.randint(0, _NUM_CLASSES, (1, 4, 4))
    logits = torch.randn(1, _NUM_CLASSES, 4, 4, requires_grad=True)
    loss_fn = SegmentationLoss(SegmentationLossConfig(name="ce"), num_classes=_NUM_CLASSES)
    loss = loss_fn(logits, targets)
    assert torch.isclose(loss, cross_entropy_loss(logits, targets, ignore_index=-100))


def test_segmentation_loss_dice() -> None:
    targets = torch.randint(0, _NUM_CLASSES, (1, 4, 4))
    logits = torch.randn(1, _NUM_CLASSES, 4, 4)
    loss_fn = SegmentationLoss(SegmentationLossConfig(name="dice"), num_classes=_NUM_CLASSES)
    loss = loss_fn(logits, targets)
    assert torch.isfinite(loss)


def test_segmentation_loss_bce() -> None:
    targets = torch.randint(0, _NUM_CLASSES, (1, 4, 4))
    logits = torch.randn(1, _NUM_CLASSES, 4, 4)
    loss_fn = SegmentationLoss(SegmentationLossConfig(name="bce"), num_classes=_NUM_CLASSES)
    loss = loss_fn(logits, targets)
    assert torch.isfinite(loss)


def test_segmentation_loss_ce_dice_combines_both_terms() -> None:
    targets = torch.randint(0, _NUM_CLASSES, (1, 4, 4))
    logits = torch.randn(1, _NUM_CLASSES, 4, 4)
    config = SegmentationLossConfig(name="ce_dice", ce_weight=0.5, dice_weight=2.0)
    loss_fn = SegmentationLoss(config, num_classes=_NUM_CLASSES)

    ce = cross_entropy_loss(logits, targets, ignore_index=-100)
    dice = DiceLoss(num_classes=_NUM_CLASSES, ignore_index=-100)(logits, targets)
    expected = 0.5 * ce + 2.0 * dice

    assert torch.isclose(loss_fn(logits, targets), expected)


def test_segmentation_loss_bce_dice_combines_both_terms() -> None:
    targets = torch.randint(0, _NUM_CLASSES, (1, 4, 4))
    logits = torch.randn(1, _NUM_CLASSES, 4, 4)
    config = SegmentationLossConfig(name="bce_dice", ce_weight=0.5, dice_weight=2.0)
    loss_fn = SegmentationLoss(config, num_classes=_NUM_CLASSES)

    bce = bce_with_logits_loss(logits, targets, num_classes=_NUM_CLASSES, ignore_index=-100)
    dice = DiceLoss(num_classes=_NUM_CLASSES, ignore_index=-100)(logits, targets)
    expected = 0.5 * bce + 2.0 * dice

    assert torch.isclose(loss_fn(logits, targets), expected)


def test_segmentation_loss_gradients_flow() -> None:
    targets = torch.randint(0, _NUM_CLASSES, (1, 4, 4))
    logits = torch.randn(1, _NUM_CLASSES, 4, 4, requires_grad=True)
    loss_fn = SegmentationLoss(SegmentationLossConfig(name="ce_dice"), num_classes=_NUM_CLASSES)
    loss_fn(logits, targets).backward()
    assert logits.grad is not None
