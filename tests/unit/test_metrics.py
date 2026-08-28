# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.tasks.segmentation.metrics.SegmentationMetric."""

import torch

from precisionai.agritune.schemas.protocols import Metric
from precisionai.agritune.tasks.segmentation.metrics import SegmentationMetric


def test_perfect_predictions_give_perfect_scores() -> None:
    metric = SegmentationMetric(num_classes=2)
    targets = torch.tensor([[[0, 1], [1, 0]]])
    metric.update(targets, targets)  # predictions == targets
    result = metric.compute()
    assert result["mean_iou"] == 1.0
    assert result["mean_dice"] == 1.0
    assert result["mean_precision"] == 1.0
    assert result["mean_recall"] == 1.0
    assert result["pixel_accuracy"] == 1.0


def test_completely_wrong_predictions_give_zero_iou_and_accuracy() -> None:
    metric = SegmentationMetric(num_classes=2)
    targets = torch.zeros(1, 2, 2, dtype=torch.long)
    predictions = torch.ones(1, 2, 2, dtype=torch.long)
    metric.update(predictions, targets)
    result = metric.compute()
    assert result["mean_iou"] == 0.0
    assert result["pixel_accuracy"] == 0.0


def test_accepts_logits_and_argmaxes_internally() -> None:
    metric_from_logits = SegmentationMetric(num_classes=2)
    metric_from_predictions = SegmentationMetric(num_classes=2)

    targets = torch.tensor([[[0, 1], [1, 0]]])
    logits = torch.zeros(1, 2, 2, 2)
    logits[:, 1, :, :] = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])  # argmax -> [[1,0],[0,1]]
    predictions = torch.tensor([[[1, 0], [0, 1]]])

    metric_from_logits.update(logits, targets)
    metric_from_predictions.update(predictions, targets)

    assert torch.equal(metric_from_logits.confusion_matrix(), metric_from_predictions.confusion_matrix())


def test_ignore_index_excludes_pixels() -> None:
    metric = SegmentationMetric(num_classes=2, ignore_index=-1)
    targets = torch.tensor([[[0, -1], [1, 0]]])
    predictions = torch.tensor([[[0, 1], [1, 0]]])  # the ignored pixel would otherwise be wrong
    metric.update(predictions, targets)
    result = metric.compute()
    assert result["mean_iou"] == 1.0


def test_multiple_updates_accumulate() -> None:
    metric = SegmentationMetric(num_classes=2)
    metric.update(torch.tensor([[[0, 0]]]), torch.tensor([[[0, 0]]]))
    metric.update(torch.tensor([[[1, 1]]]), torch.tensor([[[1, 1]]]))
    result = metric.compute()
    assert result["mean_iou"] == 1.0
    assert metric.confusion_matrix().sum().item() == 4


def test_reset_clears_accumulated_state() -> None:
    metric = SegmentationMetric(num_classes=2)
    metric.update(torch.tensor([[[1, 1]]]), torch.tensor([[[0, 0]]]))
    metric.reset()
    assert metric.confusion_matrix().sum().item() == 0


def test_absent_class_reports_zero_iou_not_nan() -> None:
    metric = SegmentationMetric(num_classes=3)  # class 2 never appears
    metric.update(torch.tensor([[[0, 1]]]), torch.tensor([[[0, 1]]]))
    result = metric.compute()
    assert result["iou_class_2"] == 0.0
    assert result["dice_class_2"] == 0.0


def test_per_class_keys_present_for_every_class() -> None:
    metric = SegmentationMetric(num_classes=3)
    metric.update(torch.tensor([[[0, 1, 2]]]), torch.tensor([[[0, 1, 2]]]))
    result = metric.compute()
    for class_index in range(3):
        assert f"iou_class_{class_index}" in result
        assert f"dice_class_{class_index}" in result
        assert f"precision_class_{class_index}" in result
        assert f"recall_class_{class_index}" in result


def test_precision_and_recall_distinguish_false_positives_from_false_negatives() -> None:
    metric = SegmentationMetric(num_classes=2)
    # class 1: 1 true positive, 1 false positive (predicted 1 but was 0), 1 false negative
    targets = torch.tensor([[[1, 0, 1]]])
    predictions = torch.tensor([[[1, 1, 0]]])
    metric.update(predictions, targets)
    result = metric.compute()
    assert result["precision_class_1"] == 0.5  # 1 true positive / 2 predicted positive
    assert result["recall_class_1"] == 0.5  # 1 true positive / 2 actual positive


def test_absent_class_reports_zero_precision_and_recall_not_nan() -> None:
    metric = SegmentationMetric(num_classes=3)  # class 2 never appears
    metric.update(torch.tensor([[[0, 1]]]), torch.tensor([[[0, 1]]]))
    result = metric.compute()
    assert result["precision_class_2"] == 0.0
    assert result["recall_class_2"] == 0.0


def test_confusion_matrix_matches_expected_counts() -> None:
    metric = SegmentationMetric(num_classes=2)
    # 3 pixels target=0 pred=0, 1 pixel target=0 pred=1, 2 pixels target=1 pred=1
    targets = torch.tensor([[[0, 0, 0, 0], [1, 1, 0, 0]]])
    predictions = torch.tensor([[[0, 0, 0, 1], [1, 1, 0, 0]]])
    metric.update(predictions, targets)
    matrix = metric.confusion_matrix()
    assert matrix[0, 0].item() == 5  # target 0, pred 0
    assert matrix[0, 1].item() == 1  # target 0, pred 1
    assert matrix[1, 1].item() == 2  # target 1, pred 1
    assert matrix[1, 0].item() == 0


def test_segmentation_metric_satisfies_metric_protocol() -> None:
    assert isinstance(SegmentationMetric(num_classes=2), Metric)
