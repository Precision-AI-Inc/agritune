# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.tasks.segmentation.task.SegmentationTask."""

import torch

from precisionai.agritune.schemas.features import EncoderFeatures
from precisionai.agritune.schemas.protocols import Task
from precisionai.agritune.tasks.segmentation.decoders.linear import LinearProbeDecoder
from precisionai.agritune.tasks.segmentation.losses import SegmentationLoss, SegmentationLossConfig
from precisionai.agritune.tasks.segmentation.task import SegmentationTask


def _features(*, batch_size: int = 2, grid: tuple[int, int] = (4, 4), patch_dim: int = 8) -> EncoderFeatures:
    height, width = grid
    return EncoderFeatures(
        patch_tokens=torch.randn(batch_size, height * width, patch_dim),
        cls_tokens=None,
        patch_grid=torch.tensor([[height, width]] * batch_size),
        valid_patch_mask=None,
        image_sizes=[(224, 224)] * batch_size,
        encoder_model="fake",
        encoder_revision=None,
    )


def test_forward_delegates_to_decoder() -> None:
    decoder = LinearProbeDecoder(patch_dim=8, num_classes=3, output_size=(16, 16))
    loss = SegmentationLoss(SegmentationLossConfig(name="ce"), num_classes=3)
    task = SegmentationTask(decoder, loss)

    outputs = task.forward(_features())
    assert outputs.shape == (2, 3, 16, 16)


def test_compute_loss_delegates_to_loss() -> None:
    decoder = LinearProbeDecoder(patch_dim=8, num_classes=3, output_size=(16, 16))
    loss = SegmentationLoss(SegmentationLossConfig(name="ce"), num_classes=3)
    task = SegmentationTask(decoder, loss)

    outputs = task.forward(_features())
    targets = torch.randint(0, 3, (2, 16, 16))
    computed = task.compute_loss(outputs, targets)
    assert torch.isfinite(computed)


def test_segmentation_task_satisfies_task_protocol() -> None:
    decoder = LinearProbeDecoder(patch_dim=8, num_classes=3, output_size=(16, 16))
    loss = SegmentationLoss(SegmentationLossConfig(name="ce"), num_classes=3)
    task = SegmentationTask(decoder, loss)
    assert isinstance(task, Task)


def test_end_to_end_forward_and_backward() -> None:
    decoder = LinearProbeDecoder(patch_dim=8, num_classes=3, output_size=(16, 16))
    loss = SegmentationLoss(SegmentationLossConfig(name="ce_dice"), num_classes=3)
    task = SegmentationTask(decoder, loss)

    outputs = task.forward(_features())
    targets = torch.randint(0, 3, (2, 16, 16))
    computed = task.compute_loss(outputs, targets)
    computed.backward()

    assert decoder.projection.weight.grad is not None
