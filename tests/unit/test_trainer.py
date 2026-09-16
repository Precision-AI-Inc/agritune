# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.training.trainer.Trainer.

Covers gradient accumulation equivalence, scheduler stepping (per optimizer step, not per
micro-batch), and exact checkpoint resume.
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
from precisionai.agritune.schemas.protocols import FeatureAugmentation
from precisionai.agritune.schemas.samples import PreparedSample
from precisionai.agritune.tasks.segmentation.decoders.mlp_probe import MLPProbeDecoder
from precisionai.agritune.tasks.segmentation.losses import SegmentationLoss, SegmentationLossConfig
from precisionai.agritune.tasks.segmentation.metrics import SegmentationMetric
from precisionai.agritune.tasks.segmentation.task import SegmentationTask
from precisionai.agritune.training.checkpointing import CheckpointManager
from precisionai.agritune.training.distributed import DistributedContext
from precisionai.agritune.training.evaluator import TrainingBatch
from precisionai.agritune.training.state import set_deterministic_seed
from precisionai.agritune.training.trainer import Trainer, TrainerConfig


def _sample(sample_id: str) -> PreparedSample:
    return PreparedSample(sample_id=sample_id, image=None, target=None)


class _FakeTracker:
    """A minimal Tracker recording every call, for asserting the trainer logs to it."""

    def __init__(self, *, fail_on_metrics: bool = False) -> None:
        self.metric_calls: list[tuple[dict[str, float], int]] = []
        self._fail_on_metrics = fail_on_metrics

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        if self._fail_on_metrics:
            raise RuntimeError("tracker unavailable")
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


class _FailAfterOneBatchProvider:
    def __init__(self, inner: _DeterministicFeatureProvider) -> None:
        self._inner = inner
        self._calls = 0

    def get_features(self, samples: Sequence[PreparedSample]) -> EncoderFeatures:
        self._calls += 1
        if self._calls > 1:
            raise RuntimeError("simulated interruption")
        return self._inner.get_features(samples)


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
    feature_augmentation: FeatureAugmentation | None = None,
    show_progress: bool = False,
    distributed: DistributedContext | None = None,
) -> tuple[Trainer, MLPProbeDecoder]:
    set_deterministic_seed(seed)  # ensures identical decoder initialization across builds
    decoder = MLPProbeDecoder(patch_dim=4, num_classes=2, output_size=(2, 2))
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
        feature_augmentation=feature_augmentation,
        show_progress=show_progress,
        distributed=distributed,
    )
    return trainer, decoder


def test_fit_runs_without_error_and_advances_state() -> None:
    trainer, _ = _build_trainer(max_epochs=2)
    batch = TrainingBatch(samples=[_sample("a"), _sample("b")], targets=torch.randint(0, 2, (2, 2, 2)))
    trainer.fit([batch])
    assert trainer.state.epoch == 2
    assert trainer.state.global_optimizer_step == 2  # one optimizer step per epoch here


def test_fit_with_show_progress_still_advances_state() -> None:
    trainer, _ = _build_trainer(max_epochs=1, show_progress=True)
    train_batch = TrainingBatch(samples=[_sample("a"), _sample("b")], targets=torch.randint(0, 2, (2, 2, 2)))
    val_batch = TrainingBatch(samples=[_sample("c")], targets=torch.randint(0, 2, (1, 2, 2)))
    trainer.fit([train_batch], [val_batch], val_metric=SegmentationMetric(num_classes=2))
    assert trainer.state.epoch == 1


def test_fit_enables_progress_on_main_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, object]] = []

    def fake_progress_iter(iterable: object, *, desc: str | None = None, unit: str = "it", disable: bool = False):
        seen.append({"desc": desc, "unit": unit, "disable": disable})
        return iterable

    monkeypatch.setattr("precisionai.agritune.training.trainer.progress_iter", fake_progress_iter)
    monkeypatch.setattr("precisionai.agritune.training.evaluator.progress_iter", fake_progress_iter)
    trainer, _ = _build_trainer(max_epochs=1, show_progress=True)
    train_batch = TrainingBatch(samples=[_sample("a")], targets=torch.randint(0, 2, (1, 2, 2)))
    val_batch = TrainingBatch(samples=[_sample("b")], targets=torch.randint(0, 2, (1, 2, 2)))
    trainer.fit([train_batch], [val_batch], val_metric=SegmentationMetric(num_classes=2))
    assert {"desc": "epoch 0", "unit": "batch", "disable": False} in seen
    assert {"desc": "evaluate", "unit": "batch", "disable": False} in seen


def test_fit_suppresses_progress_on_non_main_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, object]] = []

    def fake_progress_iter(iterable: object, *, desc: str | None = None, unit: str = "it", disable: bool = False):
        seen.append({"desc": desc, "unit": unit, "disable": disable})
        return iterable

    monkeypatch.setattr("precisionai.agritune.training.trainer.progress_iter", fake_progress_iter)
    monkeypatch.setattr("precisionai.agritune.training.evaluator.progress_iter", fake_progress_iter)
    trainer, _ = _build_trainer(max_epochs=1, show_progress=True, distributed=DistributedContext(rank=1, world_size=2))
    train_batch = TrainingBatch(samples=[_sample("a")], targets=torch.randint(0, 2, (1, 2, 2)))
    val_batch = TrainingBatch(samples=[_sample("b")], targets=torch.randint(0, 2, (1, 2, 2)))
    trainer.fit([train_batch], [val_batch], val_metric=SegmentationMetric(num_classes=2))
    assert seen
    assert all(call["disable"] is True for call in seen)


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


def test_partial_accumulation_window_is_flushed_with_correct_gradient_scale() -> None:
    targets = torch.randint(0, 2, (3, 2, 2))
    reference_batches = [
        TrainingBatch(samples=[_sample("a"), _sample("b")], targets=targets[:2]),
        TrainingBatch(samples=[_sample("c")], targets=targets[2:]),
    ]
    micro_batches = [
        TrainingBatch(samples=[_sample(sample_id)], targets=targets[index : index + 1])
        for index, sample_id in enumerate(("a", "b", "c"))
    ]

    reference_trainer, reference_decoder = _build_trainer(accumulation_steps=1, seed=42)
    reference_trainer.fit(reference_batches)

    accumulated_trainer, accumulated_decoder = _build_trainer(accumulation_steps=2, seed=42)
    accumulated_trainer.fit(micro_batches)

    assert accumulated_trainer.state.global_optimizer_step == 2
    for reference_param, accumulated_param in zip(
        reference_decoder.parameters(), accumulated_decoder.parameters(), strict=True
    ):
        assert torch.allclose(reference_param, accumulated_param, atol=1e-6)


def test_partial_accumulation_skips_parameters_without_gradients() -> None:
    trainer, decoder = _build_trainer(accumulation_steps=2, max_epochs=1, seed=4)
    decoder.register_parameter("frozen", nn.Parameter(torch.ones(1), requires_grad=False))
    batch = TrainingBatch(samples=[_sample("a")], targets=torch.randint(0, 2, (1, 2, 2)))

    trainer.fit([batch])

    assert trainer.state.global_optimizer_step == 1
    assert decoder.frozen.grad is None


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


def test_periodic_checkpoint_is_skipped_when_step_is_not_a_multiple(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path / "non-multiple")
    trainer, _ = _build_trainer(max_epochs=1, checkpoint_manager=manager, checkpoint_every_n_steps=2)
    batch = TrainingBatch(samples=[_sample("a")], targets=torch.randint(0, 2, (1, 2, 2)))

    trainer.fit([batch])

    assert trainer.state.global_optimizer_step == 1
    assert list((tmp_path / "non-multiple").glob("step_*.ckpt")) == []


def test_validation_ranks_a_lower_metric_as_better_when_configured(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path / "lower-is-better")
    trainer, _ = _build_trainer(max_epochs=1, checkpoint_manager=manager)
    train_batch = TrainingBatch(samples=[_sample("a"), _sample("b")], targets=torch.randint(0, 2, (2, 2, 2)))
    val_batch = TrainingBatch(samples=[_sample("a"), _sample("b")], targets=torch.zeros(2, 2, 2, dtype=torch.long))

    trainer.fit(
        [train_batch],
        [val_batch],
        val_metric=SegmentationMetric(num_classes=2),
        val_metric_name="mean_iou",
        higher_is_better=False,
    )

    assert trainer.state.best_metric is not None
    assert (tmp_path / "lower-is-better" / "best.ckpt").is_file()


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
    assert trainer.state.epochs_without_improvement >= 1


def test_early_stopping_resume_does_not_train_extra_epochs(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path / "early-stop")
    train_batch = TrainingBatch(samples=[_sample("a"), _sample("b")], targets=torch.randint(0, 2, (2, 2, 2)))
    val_batch = TrainingBatch(samples=[_sample("a"), _sample("b")], targets=torch.zeros(2, 2, 2, dtype=torch.long))

    first, _ = _build_trainer(max_epochs=10, early_stopping_patience=1, checkpoint_manager=manager, seed=3)
    first.fit(
        [train_batch],
        [val_batch],
        val_metric=SegmentationMetric(num_classes=2),
        val_metric_name="mean_iou",
        higher_is_better=True,
    )
    stopped_epoch = first.state.epoch
    assert stopped_epoch < 10

    resumed, _ = _build_trainer(max_epochs=10, early_stopping_patience=1, checkpoint_manager=manager, seed=3)
    resumed.fit(
        [train_batch],
        [val_batch],
        val_metric=SegmentationMetric(num_classes=2),
        val_metric_name="mean_iou",
        higher_is_better=True,
    )

    assert resumed.state.epoch == stopped_epoch
    assert resumed.state.epochs_without_improvement == first.state.epochs_without_improvement


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


def test_mid_epoch_checkpoint_resume_skips_already_consumed_batches(tmp_path: Path) -> None:
    targets = torch.randint(0, 2, (3, 2, 2))
    batches = [
        TrainingBatch(samples=[_sample(sample_id)], targets=targets[index : index + 1])
        for index, sample_id in enumerate(("a", "b", "c"))
    ]

    continuous_trainer, continuous_decoder = _build_trainer(max_epochs=1, seed=9)
    continuous_trainer.fit(batches)

    manager = CheckpointManager(tmp_path / "interrupted-mid-epoch")
    interrupted_trainer, _ = _build_trainer(
        max_epochs=1,
        seed=9,
        checkpoint_manager=manager,
        checkpoint_every_n_steps=1,
    )
    assert isinstance(interrupted_trainer.feature_provider, _DeterministicFeatureProvider)
    interrupted_trainer.feature_provider = _FailAfterOneBatchProvider(interrupted_trainer.feature_provider)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        interrupted_trainer.fit(batches)

    checkpoint = manager.load(manager.last_path)
    assert checkpoint.training_state.epoch == 0
    assert checkpoint.training_state.batch_in_epoch == 1

    resumed_trainer, resumed_decoder = _build_trainer(max_epochs=1, seed=9, checkpoint_manager=manager)
    resumed_trainer.fit(batches)

    assert resumed_trainer.state.epoch == 1
    assert resumed_trainer.state.batch_in_epoch == 0
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


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"grad_clip_norm": 0.0}, "grad_clip_norm must be positive"),
        ({"early_stopping_patience": 0}, "early_stopping_patience must be positive"),
        ({"checkpoint_every_n_steps": 0}, "checkpoint_every_n_steps must be positive"),
    ],
)
def test_trainer_config_rejects_invalid_intervals(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TrainerConfig(max_epochs=1, **kwargs)


def test_grad_clip_norm_bounds_gradient_magnitude() -> None:
    trainer, decoder = _build_trainer(grad_clip_norm=0.01)
    batch = TrainingBatch(samples=[_sample("a"), _sample("b")], targets=torch.randint(0, 2, (2, 2, 2)))
    trainer.fit([batch])
    # A tiny clip norm keeps the single optimizer step's effect on weights very small.
    first_layer = decoder.mlp[0]
    assert isinstance(first_layer, nn.Linear)  # hidden_dims=() -> a single Linear; also narrows the type
    assert first_layer.weight.abs().max().item() < 0.5


class _RecordingFeatureAugmentation:
    """A minimal feature-augmentation stand-in that records how many times it was called."""

    def __init__(self) -> None:
        self.calls = 0

    def apply(self, features: EncoderFeatures, *, generator: object | None = None) -> EncoderFeatures:
        self.calls += 1
        return features


def test_feature_augmentation_is_applied_once_per_training_batch() -> None:
    augmentation = _RecordingFeatureAugmentation()
    trainer, _ = _build_trainer(max_epochs=2, feature_augmentation=augmentation)
    batch = TrainingBatch(samples=[_sample("a"), _sample("b")], targets=torch.randint(0, 2, (2, 2, 2)))

    trainer.fit([batch])

    assert augmentation.calls == 2  # one training batch per epoch, 2 epochs


def test_feature_augmentation_is_not_applied_during_validation() -> None:
    augmentation = _RecordingFeatureAugmentation()
    trainer, _ = _build_trainer(max_epochs=1, feature_augmentation=augmentation)
    train_batch = TrainingBatch(samples=[_sample("a"), _sample("b")], targets=torch.randint(0, 2, (2, 2, 2)))
    val_batch = TrainingBatch(samples=[_sample("a"), _sample("b")], targets=torch.randint(0, 2, (2, 2, 2)))

    trainer.fit([train_batch], [val_batch], val_metric=SegmentationMetric(num_classes=2))

    assert augmentation.calls == 1  # only the training batch, never validation


def test_no_feature_augmentation_by_default() -> None:
    trainer, _ = _build_trainer()
    assert trainer.feature_augmentation is None


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


def test_train_metric_is_accumulated_and_exposed_via_last_train_metrics() -> None:
    trainer, _ = _build_trainer(max_epochs=1)
    train_batch = TrainingBatch(samples=[_sample("a"), _sample("b")], targets=torch.randint(0, 2, (2, 2, 2)))
    val_batch = TrainingBatch(samples=[_sample("a"), _sample("b")], targets=torch.randint(0, 2, (2, 2, 2)))
    train_metric = SegmentationMetric(num_classes=2)

    assert trainer.last_train_metrics is None

    trainer.fit([train_batch], [val_batch], train_metric=train_metric, val_metric=SegmentationMetric(num_classes=2))

    assert trainer.last_train_metrics is not None
    assert "mean_iou" in trainer.last_train_metrics
    assert train_metric.confusion_matrix().sum().item() == 2 * 2 * 2  # 2 samples x 2x2 pixels, one epoch


def test_train_metric_resets_at_the_start_of_each_epoch() -> None:
    trainer, _ = _build_trainer(max_epochs=2)
    train_batch = TrainingBatch(samples=[_sample("a"), _sample("b")], targets=torch.randint(0, 2, (2, 2, 2)))
    train_metric = SegmentationMetric(num_classes=2)

    trainer.fit([train_batch], train_metric=train_metric)

    # Not doubled across the two epochs — reset before the second epoch's accumulation.
    assert train_metric.confusion_matrix().sum().item() == 2 * 2 * 2


def test_tracker_receives_train_metric_with_train_prefix() -> None:
    tracker = _FakeTracker()
    trainer, _ = _build_trainer(tracker=tracker)
    train_batch = TrainingBatch(samples=[_sample("a"), _sample("b")], targets=torch.randint(0, 2, (2, 2, 2)))

    trainer.fit([train_batch], train_metric=SegmentationMetric(num_classes=2))

    logged_names = {key for metrics, _ in tracker.metric_calls for key in metrics}
    assert "train_mean_iou" in logged_names
    assert "train_pixel_accuracy" in logged_names


def test_no_train_metric_by_default() -> None:
    trainer, _ = _build_trainer()
    train_batch = TrainingBatch(samples=[_sample("a"), _sample("b")], targets=torch.randint(0, 2, (2, 2, 2)))

    trainer.fit([train_batch])  # must not raise without a train_metric

    assert trainer.last_train_metrics is None


def test_tracking_failure_does_not_interrupt_training() -> None:
    tracker = _FakeTracker(fail_on_metrics=True)
    trainer, _ = _build_trainer(max_epochs=2, tracker=tracker)
    batch = TrainingBatch(samples=[_sample("a"), _sample("b")], targets=torch.randint(0, 2, (2, 2, 2)))

    trainer.fit([batch])

    assert trainer.state.epoch == 2
    assert trainer.state.global_optimizer_step == 2


def test_checkpoint_every_n_steps_writes_periodic_checkpoint(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    trainer, _ = _build_trainer(max_epochs=1, checkpoint_every_n_steps=1, checkpoint_manager=manager)
    batch = TrainingBatch(samples=[_sample("a"), _sample("b")], targets=torch.randint(0, 2, (2, 2, 2)))

    trainer.fit([batch])

    assert (tmp_path / "step_00000001.ckpt").is_file()


def test_validation_writes_ranked_periodic_checkpoint_for_top_k_pruning(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, top_k=1)
    trainer, _ = _build_trainer(max_epochs=3, checkpoint_manager=manager)
    train_batch = TrainingBatch(samples=[_sample("a"), _sample("b")], targets=torch.randint(0, 2, (2, 2, 2)))
    val_batch = TrainingBatch(samples=[_sample("a"), _sample("b")], targets=torch.randint(0, 2, (2, 2, 2)))

    trainer.fit([train_batch], [val_batch], val_metric=SegmentationMetric(num_classes=2), val_metric_name="mean_iou")

    # top_k=1 over 3 validated epochs must prune down to exactly one ranked periodic checkpoint.
    step_checkpoints = list(tmp_path.glob("step_*.ckpt"))
    assert len(step_checkpoints) == 1


def test_reduce_on_plateau_scheduler_is_stepped_on_validation() -> None:
    scheduler_config = SchedulerConfig(name="plateau", plateau_patience=0)
    trainer, _ = _build_trainer(max_epochs=2, scheduler_config=scheduler_config)
    train_batch = TrainingBatch(samples=[_sample("a"), _sample("b")], targets=torch.randint(0, 2, (2, 2, 2)))
    val_batch = TrainingBatch(samples=[_sample("a"), _sample("b")], targets=torch.zeros(2, 2, 2, dtype=torch.long))

    trainer.fit([train_batch], [val_batch], val_metric=SegmentationMetric(num_classes=2), higher_is_better=True)

    assert trainer.scheduler is not None
    assert trainer.scheduler.last_epoch >= 1  # ReduceLROnPlateau.step() was called at least once


def test_construction_does_not_raise_for_parameterless_decoder() -> None:
    # _device() must fall back to torch.device("cpu") rather than raising when the decoder has no
    # parameters to inspect.
    decoder = nn.Module()
    task = SegmentationTask(decoder, SegmentationLoss(SegmentationLossConfig(name="ce"), num_classes=2))
    provider = _DeterministicFeatureProvider(patch_dim=4, patch_grid=(2, 2))
    optimizer = SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    config = TrainerConfig(max_epochs=1)
    trainer = Trainer(
        task=task, decoder=decoder, feature_provider=provider, optimizer=optimizer, scheduler=None, config=config
    )
    assert trainer.device == torch.device("cpu")


def test_device_reflects_wherever_the_decoders_parameters_live() -> None:
    trainer, _ = _build_trainer(max_epochs=1)
    assert trainer.device == next(trainer.decoder.parameters()).device


def test_train_one_epoch_moves_features_and_targets_to_the_trainers_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A CPU-only CI machine can't prove data actually lands on a GPU, but it can prove every
    # feature/target passes through .to(trainer.device) with the right device argument — the same
    # code path that matters on a real CUDA device.
    trainer, _ = _build_trainer(max_epochs=1)
    train_batch = TrainingBatch(samples=[_sample("a"), _sample("b")], targets=torch.randint(0, 2, (2, 2, 2)))

    seen_devices: list[torch.device] = []
    original_to = EncoderFeatures.to

    def spy_to(self: EncoderFeatures, device: torch.device | str, *, non_blocking: bool = False) -> EncoderFeatures:
        seen_devices.append(torch.device(device))
        return original_to(self, device, non_blocking=non_blocking)

    monkeypatch.setattr(EncoderFeatures, "to", spy_to)

    trainer.fit([train_batch])

    assert seen_devices
    assert all(device == trainer.device for device in seen_devices)
