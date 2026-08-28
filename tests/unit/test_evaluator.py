# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.training.evaluator."""

import torch

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
