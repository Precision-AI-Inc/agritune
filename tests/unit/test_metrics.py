# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.tasks.segmentation.metrics.SegmentationMetric."""

import pytest
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


def test_update_rejects_out_of_range_targets() -> None:
    metric = SegmentationMetric(num_classes=2)
    with pytest.raises(ValueError, match=r"targets contain labels outside \[0, 2\)"):
        metric.update(torch.tensor([[[0, 1]]]), torch.tensor([[[0, 2]]]))


def test_update_rejects_out_of_range_predictions() -> None:
    metric = SegmentationMetric(num_classes=2)
    with pytest.raises(ValueError, match=r"predictions contain labels outside \[0, 2\)"):
        metric.update(torch.tensor([[[0, 2]]]), torch.tensor([[[0, 1]]]))


def test_update_rejects_wrong_output_rank() -> None:
    metric = SegmentationMetric(num_classes=2)
    with pytest.raises(ValueError, match="outputs must have shape"):
        metric.update(torch.zeros(2, 2), torch.zeros(1, 2, 2, dtype=torch.long))


def test_update_rejects_shape_mismatch() -> None:
    metric = SegmentationMetric(num_classes=2)
    with pytest.raises(ValueError, match="predictions and targets must share shape"):
        metric.update(torch.zeros(1, 2, 2, dtype=torch.long), torch.zeros(1, 3, 3, dtype=torch.long))


def test_update_resizes_mismatched_logits_to_match_targets() -> None:
    metric = SegmentationMetric(num_classes=2)
    targets = torch.zeros(1, 2, 2, dtype=torch.long)
    mismatched_logits = torch.zeros(1, 2, 4, 4)
    mismatched_logits[:, 0] = 10.0  # confidently predicts class 0 everywhere, at 4x4 resolution

    metric.update(mismatched_logits, targets)  # must not raise despite the spatial size mismatch

    result = metric.compute()
    assert result["pixel_accuracy"] == 1.0  # class 0 everywhere matches an all-zero target
    assert result["iou_class_0"] == 1.0


def test_update_rejects_wrong_logit_class_dimension() -> None:
    metric = SegmentationMetric(num_classes=2)
    with pytest.raises(ValueError, match="logits class dimension must be 2"):
        metric.update(torch.zeros(1, 3, 2, 2), torch.zeros(1, 2, 2, dtype=torch.long))


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA-enabled machine")
def test_update_follows_a_cuda_devices_outputs_and_targets() -> None:
    # The confusion matrix starts on CPU (see __init__); update() must migrate it to wherever
    # outputs/targets actually are instead of raising a device-mismatch error, since Trainer moves
    # every batch to a CUDA device before calling Metric.update.
    metric = SegmentationMetric(num_classes=2)
    targets = torch.tensor([[[0, 1], [1, 0]]], device="cuda")

    metric.update(targets, targets)

    assert metric.confusion_matrix().device.type == "cuda"
    result = metric.compute()
    assert result["mean_iou"] == 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA-enabled machine")
def test_update_stays_on_cuda_across_multiple_batches() -> None:
    metric = SegmentationMetric(num_classes=2)
    targets = torch.tensor([[[0, 1], [1, 0]]], device="cuda")

    metric.update(targets, targets)
    metric.update(targets, targets)

    assert metric.confusion_matrix().device.type == "cuda"
    assert metric.confusion_matrix().sum().item() == 8  # 2 batches x 4 pixels
