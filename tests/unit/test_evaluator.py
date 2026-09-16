# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.training.evaluator."""

import pytest
import torch

from precisionai.agritune.schemas.features import EncoderFeatures
from precisionai.agritune.schemas.samples import PreparedSample
from precisionai.agritune.tasks.segmentation.decoders.mlp_probe import MLPProbeDecoder
from precisionai.agritune.tasks.segmentation.losses import SegmentationLoss, SegmentationLossConfig
from precisionai.agritune.tasks.segmentation.metrics import SegmentationMetric
from precisionai.agritune.tasks.segmentation.task import SegmentationTask
from precisionai.agritune.training.evaluator import TrainingBatch, evaluate
from tests.fixtures.fake_pipeline import FakeFeatureProvider


def _sample(sample_id: str) -> PreparedSample:
    return PreparedSample(sample_id=sample_id, image=None, target=None)


def _task() -> SegmentationTask:
    decoder = MLPProbeDecoder(patch_dim=8, num_classes=3, output_size=(4, 4))
    loss = SegmentationLoss(SegmentationLossConfig(name="ce"), num_classes=3)
    return SegmentationTask(decoder, loss)


def test_evaluate_accumulates_across_all_batches() -> None:
    task = _task()
    provider = FakeFeatureProvider(patch_dim=8, cls_dim=None, patch_grid=(2, 2))
    metric = SegmentationMetric(num_classes=3)

    batches = [
        TrainingBatch(samples=[_sample("a"), _sample("b")], targets=torch.randint(0, 3, (2, 4, 4))) for _ in range(3)
    ]
    result = evaluate(task, provider, batches, metric)

    assert "mean_iou" in result
    assert metric.confusion_matrix().sum().item() == 3 * 2 * 4 * 4  # 3 batches x 2 samples x 4x4 pixels


def test_evaluate_with_show_progress_still_computes_metrics() -> None:
    task = _task()
    provider = FakeFeatureProvider(patch_dim=8, cls_dim=None, patch_grid=(2, 2))
    metric = SegmentationMetric(num_classes=3)

    batches = [TrainingBatch(samples=[_sample("a")], targets=torch.randint(0, 3, (1, 4, 4)))]
    result = evaluate(task, provider, batches, metric, show_progress=True)

    assert "mean_iou" in result


def test_evaluate_disables_progress_bar_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_progress_iter(iterable: object, *, desc: str | None = None, unit: str = "it", disable: bool = False):
        captured.update({"desc": desc, "unit": unit, "disable": disable})
        return iterable

    monkeypatch.setattr("precisionai.agritune.training.evaluator.progress_iter", fake_progress_iter)
    batches = [TrainingBatch(samples=[_sample("a")], targets=torch.randint(0, 3, (1, 4, 4)))]
    evaluate(
        _task(),
        FakeFeatureProvider(patch_dim=8, cls_dim=None, patch_grid=(2, 2)),
        batches,
        SegmentationMetric(num_classes=3),
    )
    assert captured == {"desc": "evaluate", "unit": "batch", "disable": True}


def test_evaluate_enables_progress_bar_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_progress_iter(iterable: object, *, desc: str | None = None, unit: str = "it", disable: bool = False):
        captured.update({"desc": desc, "unit": unit, "disable": disable})
        return iterable

    monkeypatch.setattr("precisionai.agritune.training.evaluator.progress_iter", fake_progress_iter)
    batches = [TrainingBatch(samples=[_sample("a")], targets=torch.randint(0, 3, (1, 4, 4)))]
    evaluate(
        _task(),
        FakeFeatureProvider(patch_dim=8, cls_dim=None, patch_grid=(2, 2)),
        batches,
        SegmentationMetric(num_classes=3),
        show_progress=True,
    )
    assert captured == {"desc": "evaluate", "unit": "batch", "disable": False}


def test_evaluate_reports_loss_weighted_by_batch_size() -> None:
    task = _task()
    provider = FakeFeatureProvider(patch_dim=8, cls_dim=None, patch_grid=(2, 2))
    metric = SegmentationMetric(num_classes=3)

    small_batch = TrainingBatch(samples=[_sample("a")], targets=torch.randint(0, 3, (1, 4, 4)))
    large_batch = TrainingBatch(samples=[_sample("b"), _sample("c")], targets=torch.randint(0, 3, (2, 4, 4)))

    result = evaluate(task, provider, [small_batch, large_batch], metric)

    features_a = provider.get_features(small_batch.samples)
    features_b = provider.get_features(large_batch.samples)
    loss_a = task.compute_loss(task.forward(features_a), small_batch.targets).item()
    loss_b = task.compute_loss(task.forward(features_b), large_batch.targets).item()
    expected = (loss_a * 1 + loss_b * 2) / 3

    assert "loss" in result
    assert result["loss"] == pytest.approx(expected, rel=1e-4)


def test_evaluate_loss_is_zero_for_no_batches() -> None:
    task = _task()
    provider = FakeFeatureProvider(patch_dim=8, cls_dim=None, patch_grid=(2, 2))
    metric = SegmentationMetric(num_classes=3)

    result = evaluate(task, provider, [], metric)

    assert result["loss"] == 0.0


def test_evaluate_does_not_track_gradients() -> None:
    task = _task()
    provider = FakeFeatureProvider(patch_dim=8, cls_dim=None, patch_grid=(2, 2))
    metric = SegmentationMetric(num_classes=3)
    batch = TrainingBatch(samples=[_sample("a")], targets=torch.randint(0, 3, (1, 4, 4)))

    evaluate(task, provider, [batch], metric)

    mlp_decoder = task.decoder
    assert isinstance(mlp_decoder, MLPProbeDecoder)
    assert mlp_decoder.mlp[0].weight.grad is None


def test_training_batch_holds_samples_and_targets() -> None:
    targets = torch.zeros(1, 2, 2, dtype=torch.long)
    sample = _sample("s1")
    batch = TrainingBatch(samples=[sample], targets=targets)
    assert batch.samples == [sample]
    assert torch.equal(batch.targets, targets)


def test_evaluate_with_no_device_leaves_features_and_targets_where_they_already_are(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_devices: list[torch.device] = []
    original_to = EncoderFeatures.to

    def spy_to(self: EncoderFeatures, device: torch.device | str, *, non_blocking: bool = False) -> EncoderFeatures:
        seen_devices.append(torch.device(device))
        return original_to(self, device, non_blocking=non_blocking)

    monkeypatch.setattr(EncoderFeatures, "to", spy_to)
    task = _task()
    provider = FakeFeatureProvider(patch_dim=8, cls_dim=None, patch_grid=(2, 2))
    batch = TrainingBatch(samples=[_sample("a")], targets=torch.randint(0, 3, (1, 4, 4)))

    evaluate(task, provider, [batch], SegmentationMetric(num_classes=3))

    assert seen_devices == [torch.device("cpu")]


def test_evaluate_moves_features_and_targets_to_the_given_device(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_devices: list[torch.device] = []
    original_to = EncoderFeatures.to

    def spy_to(self: EncoderFeatures, device: torch.device | str, *, non_blocking: bool = False) -> EncoderFeatures:
        seen_devices.append(torch.device(device))
        return original_to(self, device, non_blocking=non_blocking)

    monkeypatch.setattr(EncoderFeatures, "to", spy_to)
    task = _task()
    provider = FakeFeatureProvider(patch_dim=8, cls_dim=None, patch_grid=(2, 2))
    batch = TrainingBatch(samples=[_sample("a")], targets=torch.randint(0, 3, (1, 4, 4)))

    result = evaluate(task, provider, [batch], SegmentationMetric(num_classes=3), device=torch.device("cpu"))

    assert "mean_iou" in result
    assert seen_devices == [torch.device("cpu")]
