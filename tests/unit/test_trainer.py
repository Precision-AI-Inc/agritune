# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.training.trainer.Trainer.

Covers definition-of-done items from ``agritune_implementation_plan.md`` §22: gradient
accumulation equivalence, scheduler stepping (per optimizer step, not per micro-batch), and exact
checkpoint resume.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from torch.optim import SGD

from precisionai.agritune.optimization.schedulers import SchedulerConfig, build_scheduler
from precisionai.agritune.schemas.features import EncoderFeatures
from precisionai.agritune.schemas.samples import PreparedSample
from precisionai.agritune.tasks.segmentation.decoders.linear import LinearProbeDecoder
from precisionai.agritune.tasks.segmentation.losses import SegmentationLoss, SegmentationLossConfig
from precisionai.agritune.tasks.segmentation.metrics import SegmentationMetric
from precisionai.agritune.tasks.segmentation.task import SegmentationTask
from precisionai.agritune.training.checkpointing import CheckpointManager
from precisionai.agritune.training.evaluator import TrainingBatch
from precisionai.agritune.training.state import set_deterministic_seed
from precisionai.agritune.training.trainer import Trainer, TrainerConfig


def _sample(sample_id: str) -> PreparedSample:
    return PreparedSample(sample_id=sample_id, image=None, target=None)


class _FakeTracker:
    """A minimal Tracker recording every call, for asserting the trainer logs to it."""

    def __init__(self) -> None:
        self.metric_calls: list[tuple[dict[str, float], int]] = []

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        self.metric_calls.append((metrics, step))

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_artifact(self, path: str) -> None:
        pass

    def close(self) -> None:
        pass


class _DeterministicFeatureProvider:
    """Maps each distinct sample (by ``sample_id``) to a fixed feature vector, regardless of what
    batch it's grouped into — lets tests compare different batchings of the same samples."""

    def __init__(self, *, patch_dim: int, patch_grid: tuple[int, int]) -> None:
        self._patch_dim = patch_dim
        self._patch_grid = patch_grid

    def get_features(self, samples: Sequence[PreparedSample]) -> EncoderFeatures:
        height, width = self._patch_grid
        num_patches = height * width
        rows = []
        for sample in samples:
            seed = abs(hash(sample.sample_id)) % (2**31)
            generator = torch.Generator().manual_seed(seed)
            rows.append(torch.randn(1, num_patches, self._patch_dim, generator=generator))
        batch_size = len(samples)
        return EncoderFeatures(
            patch_tokens=torch.cat(rows, dim=0),
            cls_tokens=None,
            patch_grid=torch.tensor([[height, width]] * batch_size),
            valid_patch_mask=None,
            image_sizes=[(224, 224)] * batch_size,
            encoder_model="deterministic-fake",
            encoder_revision=None,
        )


def _build_trainer(
    *,
    accumulation_steps: int = 1,
    max_epochs: int = 1,
    checkpoint_manager: CheckpointManager | None = None,
    seed: int = 0,
    scheduler_config: SchedulerConfig | None = None,
    early_stopping_patience: int | None = None,
    grad_clip_norm: float | None = None,
    checkpoint_every_n_steps: int | None = None,
    tracker: _FakeTracker | None = None,
) -> tuple[Trainer, LinearProbeDecoder]:
    set_deterministic_seed(seed)  # ensures identical decoder initialization across builds
    decoder = LinearProbeDecoder(patch_dim=4, num_classes=2, output_size=(2, 2))
    loss = SegmentationLoss(SegmentationLossConfig(name="ce"), num_classes=2)
    task = SegmentationTask(decoder, loss)
    provider = _DeterministicFeatureProvider(patch_dim=4, patch_grid=(2, 2))
    optimizer = SGD(decoder.parameters(), lr=0.1)
    scheduler = build_scheduler(optimizer, scheduler_config) if scheduler_config is not None else None
    config = TrainerConfig(
        max_epochs=max_epochs,
        accumulation_steps=accumulation_steps,
        seed=seed,
        grad_clip_norm=grad_clip_norm,
        checkpoint_every_n_steps=checkpoint_every_n_steps,
        early_stopping_patience=early_stopping_patience,
        fingerprints={"encoder": "deterministic-fake"},
    )
    trainer = Trainer(
        task=task,
        decoder=decoder,
        feature_provider=provider,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        checkpoint_manager=checkpoint_manager,
        tracker=tracker,
    )
    return trainer, decoder


def test_fit_runs_without_error_and_advances_state() -> None:
    trainer, _ = _build_trainer(max_epochs=2)
    batch = TrainingBatch(samples=[_sample("a"), _sample("b")], targets=torch.randint(0, 2, (2, 2, 2)))
    trainer.fit([batch])
    assert trainer.state.epoch == 2
    assert trainer.state.global_optimizer_step == 2  # one optimizer step per epoch here


def test_gradient_accumulation_equivalence() -> None:
    targets = torch.randint(0, 2, (2, 2, 2))
    combined_batch = TrainingBatch(samples=[_sample("a"), _sample("b")], targets=targets)
    micro_batches = [
        TrainingBatch(samples=[_sample("a")], targets=targets[0:1]),
        TrainingBatch(samples=[_sample("b")], targets=targets[1:2]),
    ]

    trainer_combined, decoder_combined = _build_trainer(accumulation_steps=1, seed=42)
    trainer_combined.fit([combined_batch])

    trainer_accum, decoder_accum = _build_trainer(accumulation_steps=2, seed=42)
    trainer_accum.fit(micro_batches)

    for combined_param, accum_param in zip(decoder_combined.parameters(), decoder_accum.parameters(), strict=True):
        assert torch.allclose(combined_param, accum_param, atol=1e-6)


def test_scheduler_steps_once_per_optimizer_step_not_per_micro_batch() -> None:
    scheduler_config = SchedulerConfig(name="linear_warmup", total_steps=4)
    trainer, _ = _build_trainer(accumulation_steps=2, max_epochs=1, scheduler_config=scheduler_config)
    targets = torch.randint(0, 2, (4, 2, 2))
    # 4 micro-batches, accumulation_steps=2 -> only 2 optimizer steps, so only 2 scheduler steps.
    micro_batches = [TrainingBatch(samples=[_sample(f"s{i}")], targets=targets[i : i + 1]) for i in range(4)]

    trainer.fit(micro_batches)

    assert trainer.state.global_optimizer_step == 2
    assert trainer.scheduler is not None
    assert trainer.scheduler.last_epoch == 2


def test_early_stopping_triggers_after_patience_exceeded() -> None:
    trainer, _ = _build_trainer(max_epochs=10, early_stopping_patience=1)
    train_batch = TrainingBatch(samples=[_sample("a"), _sample("b")], targets=torch.randint(0, 2, (2, 2, 2)))
    val_batch = TrainingBatch(samples=[_sample("a"), _sample("b")], targets=torch.zeros(2, 2, 2, dtype=torch.long))

    trainer.fit(
        [train_batch],
        [val_batch],
        val_metric=SegmentationMetric(num_classes=2),
        val_metric_name="mean_iou",
        higher_is_better=True,
    )

    # Deterministic features + a fixed val target mean the metric can improve at most once,
    # then plateaus — with patience=1, training must stop well before 10 epochs.
    assert trainer.state.epoch < 10


def test_exact_checkpoint_resume(tmp_path: Path) -> None:
    targets = torch.randint(0, 2, (2, 2, 2))
    batch = TrainingBatch(samples=[_sample("a"), _sample("b")], targets=targets)

    continuous_dir = tmp_path / "continuous"
    continuous_manager = CheckpointManager(continuous_dir)
    continuous_trainer, continuous_decoder = _build_trainer(max_epochs=4, seed=7, checkpoint_manager=continuous_manager)
    continuous_trainer.fit([batch])

    interrupted_dir = tmp_path / "interrupted"
    interrupted_manager = CheckpointManager(interrupted_dir)
    first_half_trainer, _ = _build_trainer(max_epochs=2, seed=7, checkpoint_manager=interrupted_manager)
    first_half_trainer.fit([batch])

    second_half_trainer, resumed_decoder = _build_trainer(max_epochs=4, seed=7, checkpoint_manager=interrupted_manager)
    second_half_trainer.fit([batch])

    assert second_half_trainer.state.epoch == 4
    assert second_half_trainer.state.global_optimizer_step == continuous_trainer.state.global_optimizer_step
    for continuous_param, resumed_param in zip(
        continuous_decoder.parameters(), resumed_decoder.parameters(), strict=True
    ):
        assert torch.allclose(continuous_param, resumed_param, atol=1e-6)


def test_resume_restores_scheduler_state(tmp_path: Path) -> None:
    scheduler_config = SchedulerConfig(name="linear_warmup", total_steps=10)
    batch = TrainingBatch(samples=[_sample("a"), _sample("b")], targets=torch.randint(0, 2, (2, 2, 2)))
    manager = CheckpointManager(tmp_path)

    first_trainer, _ = _build_trainer(max_epochs=2, checkpoint_manager=manager, scheduler_config=scheduler_config)
    first_trainer.fit([batch])
    steps_after_first_run = first_trainer.state.global_optimizer_step

    second_trainer, _ = _build_trainer(max_epochs=4, checkpoint_manager=manager, scheduler_config=scheduler_config)
    assert second_trainer.scheduler is not None
    assert second_trainer.scheduler.last_epoch == steps_after_first_run


def test_invalid_accumulation_steps_raises() -> None:
    with pytest.raises(ValueError, match="accumulation_steps must be >= 1"):
        TrainerConfig(max_epochs=1, accumulation_steps=0)


def test_grad_clip_norm_bounds_gradient_magnitude() -> None:
    trainer, decoder = _build_trainer(grad_clip_norm=0.01)
    batch = TrainingBatch(samples=[_sample("a"), _sample("b")], targets=torch.randint(0, 2, (2, 2, 2)))
    trainer.fit([batch])
    # A tiny clip norm keeps the single optimizer step's effect on weights very small.
    assert decoder.projection.weight.abs().max().item() < 0.5


def test_tracker_receives_train_and_val_metrics() -> None:
    tracker = _FakeTracker()
    trainer, _ = _build_trainer(tracker=tracker)
    train_batch = TrainingBatch(samples=[_sample("a"), _sample("b")], targets=torch.randint(0, 2, (2, 2, 2)))
    val_batch = TrainingBatch(samples=[_sample("a"), _sample("b")], targets=torch.randint(0, 2, (2, 2, 2)))

    trainer.fit([train_batch], [val_batch], val_metric=SegmentationMetric(num_classes=2))

    train_metric_names = {key for metrics, _ in tracker.metric_calls for key in metrics if not key.startswith("val_")}
    val_metric_names = {key for metrics, _ in tracker.metric_calls for key in metrics if key.startswith("val_")}
    assert "train_loss" in train_metric_names
    assert any(val_metric_names)


def test_checkpoint_every_n_steps_writes_periodic_checkpoint(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    trainer, _ = _build_trainer(max_epochs=1, checkpoint_every_n_steps=1, checkpoint_manager=manager)
    batch = TrainingBatch(samples=[_sample("a"), _sample("b")], targets=torch.randint(0, 2, (2, 2, 2)))

    trainer.fit([batch])

    assert (tmp_path / "step_00000001.ckpt").is_file()


def test_reduce_on_plateau_scheduler_is_stepped_on_validation() -> None:
    scheduler_config = SchedulerConfig(name="plateau", plateau_patience=0)
    trainer, _ = _build_trainer(max_epochs=2, scheduler_config=scheduler_config)
    train_batch = TrainingBatch(samples=[_sample("a"), _sample("b")], targets=torch.randint(0, 2, (2, 2, 2)))
    val_batch = TrainingBatch(samples=[_sample("a"), _sample("b")], targets=torch.zeros(2, 2, 2, dtype=torch.long))

    trainer.fit([train_batch], [val_batch], val_metric=SegmentationMetric(num_classes=2), higher_is_better=True)

    assert trainer.scheduler is not None
    assert trainer.scheduler.last_epoch >= 1  # ReduceLROnPlateau.step() was called at least once


def test_construction_does_not_raise_for_parameterless_decoder() -> None:
    # _device_type() must fall back to "cpu" rather than raising when the decoder has no
    # parameters to inspect.
    decoder = nn.Module()
    task = SegmentationTask(decoder, SegmentationLoss(SegmentationLossConfig(name="ce"), num_classes=2))
    provider = _DeterministicFeatureProvider(patch_dim=4, patch_grid=(2, 2))
    optimizer = SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    config = TrainerConfig(max_epochs=1)
    Trainer(task=task, decoder=decoder, feature_provider=provider, optimizer=optimizer, scheduler=None, config=config)
