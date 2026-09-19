# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""``Trainer`` — orchestrates training; contains no task-specific logic.

Pseudo-flow::

    for batch in loader:
        features = feature_provider.get_features(batch.samples)
        outputs = task.forward(features)
        loss = task.compute_loss(outputs, batch.targets)
        loss /= accumulation_steps
        backward(loss)
        if optimizer_step:
            clip_gradients()
            optimizer.step()
            scheduler.step()

``batches`` must be a true re-iterable (a ``torch.utils.data.DataLoader`` or a list, not a
one-shot generator) — the trainer iterates it once per epoch, calling ``iter()`` on it fresh
each time via a plain ``for`` loop.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from precisionai.agritune.logging import get_logger, progress_iter
from precisionai.agritune.schemas.protocols import FeatureAugmentation, FeatureProvider, Metric, Task, Tracker
from precisionai.agritune.training.checkpointing import Checkpoint, CheckpointManager
from precisionai.agritune.training.distributed import DistributedContext
from precisionai.agritune.training.evaluator import TrainingBatch, evaluate
from precisionai.agritune.training.precision import PrecisionConfig, PrecisionContext
from precisionai.agritune.training.prefetch import PrefetchingFeatureLoader
from precisionai.agritune.training.state import (
    TrainingState,
    capture_rng_state,
    restore_rng_state,
    set_deterministic_seed,
)

logger = get_logger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for :class:`Trainer`.

    Attributes
    ----------
    max_epochs : int
        Number of epochs to train for.
    accumulation_steps : int
        Micro-batches accumulated before each optimizer step.
    grad_clip_norm : float | None
        Max gradient norm for clipping; ``None`` disables clipping.
    precision : PrecisionConfig
        Mixed-precision configuration.
    seed : int
        Deterministic seed applied at the start of :meth:`Trainer.fit`.
    early_stopping_patience : int | None
        Stop after this many validation passes with no improvement; ``None`` disables early
        stopping.
    checkpoint_every_n_steps : int | None
        Also write a periodic checkpoint every this many optimizer steps; ``None`` disables it
        (only end-of-epoch/best checkpoints are written).
    fingerprints : dict[str, str]
        Identifies this run's configuration/dataset/encoder, stored in every checkpoint and
        checked on resume — see :class:`~precisionai.agritune.training.checkpointing.CheckpointManager`.
    """

    max_epochs: int
    accumulation_steps: int = 1
    grad_clip_norm: float | None = None
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    seed: int = 0
    early_stopping_patience: int | None = None
    checkpoint_every_n_steps: int | None = None
    fingerprints: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate optimizer-step, clipping, stopping, and checkpoint intervals."""
        if self.accumulation_steps < 1:
            raise ValueError(f"accumulation_steps must be >= 1; got {self.accumulation_steps}")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError(f"grad_clip_norm must be positive; got {self.grad_clip_norm}")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError(f"early_stopping_patience must be positive; got {self.early_stopping_patience}")
        if self.checkpoint_every_n_steps is not None and self.checkpoint_every_n_steps < 1:
            raise ValueError(f"checkpoint_every_n_steps must be positive; got {self.checkpoint_every_n_steps}")


class Trainer:
    """Orchestrates training over a :class:`Task`, via a :class:`FeatureProvider`.

    Never imports or calls anything in ``precisionai.agritune.encoder`` — see CLAUDE.md's
    architectural rule.

    Parameters
    ----------
    task : Task
        The task being trained (binds a decoder to a loss).
    decoder : torch.nn.Module
        The trainable module — also ``task``'s decoder; passed separately since ``Task`` is a
        protocol and does not guarantee ``.parameters()``/``.train()``/``.eval()``.
    feature_provider : FeatureProvider
        Supplies features for each batch.
    optimizer : torch.optim.Optimizer
    scheduler : LRScheduler | ReduceLROnPlateau | None
        Stepped once per optimizer step (``LRScheduler``) or once per validation with the
        monitored metric (``ReduceLROnPlateau``).
    config : TrainerConfig
    checkpoint_manager : CheckpointManager | None, optional
        When given, resumes from ``last.ckpt`` if present, and checkpoints during
        :meth:`fit`.
    tracker : Tracker | None, optional
        Receives per-step and per-epoch metrics.
    distributed : DistributedContext | None, optional
        Defaults to a single-process context; only the main process checkpoints or logs to the
        tracker.
    feature_augmentation : FeatureAugmentation | None, optional
        When given, applied to every training batch's features right after
        ``feature_provider.get_features`` and before ``task.forward`` — never during validation,
        matching the usual train-only role of augmentation. ``None`` disables it entirely.
    show_progress : bool, optional
        Render a ``tqdm`` bar over each epoch's training batches (and the validation pass); always
        suppressed on non-main processes. Defaults to ``False`` so headless callers (e.g. the API)
        see no terminal output.

    Attributes
    ----------
    last_train_metrics : dict[str, float] | None
        Whatever ``train_metric.compute()`` (see :meth:`fit`) returned for the most recently
        completed training epoch — ``None`` before the first one, or if ``train_metric`` was
        never given to :meth:`fit`.
    last_val_metrics : dict[str, float] | None
        Whatever :func:`~precisionai.agritune.training.evaluator.evaluate` returned for the most
        recent validation pass (including its ``"loss"`` entry) — ``None`` before the first one.
    """

    def __init__(
        self,
        *,
        task: Task,
        decoder: nn.Module,
        feature_provider: FeatureProvider,
        optimizer: Optimizer,
        scheduler: LRScheduler | ReduceLROnPlateau | None,
        config: TrainerConfig,
        checkpoint_manager: CheckpointManager | None = None,
        tracker: Tracker | None = None,
        distributed: DistributedContext | None = None,
        feature_augmentation: FeatureAugmentation | None = None,
        show_progress: bool = False,
    ) -> None:
        self.task = task
        self.decoder = decoder
        self.feature_provider = feature_provider
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.tracker = tracker
        self.distributed = distributed or DistributedContext()
        self.feature_augmentation = feature_augmentation
        self.show_progress = show_progress
        self.device = self._device()
        self.precision = PrecisionContext(config.precision, device_type=self.device.type)
        self.state = TrainingState()
        self.last_train_metrics: dict[str, float] | None = None
        self.last_val_metrics: dict[str, float] | None = None
        self._resumed = False

        if self.checkpoint_manager is not None and self.checkpoint_manager.has_last():
            self._resume()
            self._resumed = True

    def fit(
        self,
        train_batches: Iterable[TrainingBatch],
        val_batches: Iterable[TrainingBatch] | None = None,
        *,
        train_metric: Metric | None = None,
        val_metric: Metric | None = None,
        val_metric_name: str = "mean_iou",
        higher_is_better: bool = True,
    ) -> None:
        """Train for ``config.max_epochs`` (remaining) epochs, optionally validating each epoch.

        Parameters
        ----------
        train_batches : Iterable[TrainingBatch]
            Re-iterated once per epoch.
        val_batches : Iterable[TrainingBatch] | None, optional
            Re-iterated once per epoch for validation, if given together with ``val_metric``.
        train_metric : Metric | None, optional
            Accumulates statistics over every training batch each epoch (reset at the start of
            the epoch) and is logged as ``train_{name}`` alongside ``train_loss``. Computed from
            the model in ``.train()`` mode on (possibly augmented) training batches, so it is
            noisier and less comparable across epochs than the equivalent validation metric —
            useful mainly for spotting a train/val gap, not as a model-selection signal.
        val_metric : Metric | None, optional
            Accumulates validation statistics; reset at the start of each validation pass.
        val_metric_name : str, optional
            Key into ``val_metric.compute()`` used for best-checkpoint tracking, the scheduler
            (if ``ReduceLROnPlateau``), and early stopping.
        higher_is_better : bool, optional
            Whether a larger ``val_metric_name`` value is better (true for mIoU/Dice/accuracy;
            set ``False`` for a loss-like metric).

        Raises
        ------
        ValueError
            If ``self.scheduler`` is a ``ReduceLROnPlateau`` but ``val_batches``/``val_metric``
            are not both given — it is only ever stepped from a validation pass (see
            ``_after_validation``), so it would otherwise silently never step for the entire run.
        """
        if isinstance(self.scheduler, ReduceLROnPlateau) and (val_batches is None or val_metric is None):
            raise ValueError(
                "scheduler is ReduceLROnPlateau, which only ever steps after a validation pass — "
                "fit() must be called with both val_batches and val_metric, or the learning rate "
                "would silently never change for the entire run"
            )
        if not self._resumed:
            set_deterministic_seed(self.config.seed)
        for epoch in range(self.state.epoch, self.config.max_epochs):
            if self._should_stop_early():
                logger.info("early stopping before epoch %d", epoch)
                break
            self._train_one_epoch(train_batches, train_metric)
            if train_metric is not None:
                self._after_train_epoch(train_metric)

            if val_batches is not None and val_metric is not None:
                val_metric.reset()
                self.decoder.eval()  # restored to train() by the next epoch's _train_one_epoch
                metrics = evaluate(
                    self.task,
                    self.feature_provider,
                    val_batches,
                    val_metric,
                    device=self.device,
                    show_progress=self.show_progress and self.distributed.is_main_process,
                )
                self.last_val_metrics = metrics
                # Advances state to epoch + 1 itself (see its own docstring for why), so no
                # further checkpoint write is needed here in this branch.
                self._after_validation(metrics, val_metric_name, higher_is_better, next_epoch=epoch + 1)
            else:
                self.state.epoch = epoch + 1
                self.state.batch_in_epoch = 0
                self._checkpoint(periodic=False, is_best=False, metric_value=None)

            if self._should_stop_early():
                logger.info("early stopping after epoch %d", epoch)
                break

    def _train_one_epoch(self, batches: Iterable[TrainingBatch], train_metric: Metric | None) -> None:
        self.decoder.train()
        if train_metric is not None:
            train_metric.reset()
        accumulated_steps = 0
        last_loss_value = 0.0
        resume_offset = self.state.batch_in_epoch
        non_blocking = self.device.type == "cuda"

        # Skip already-trained batches on the raw iterator, before wrapping it in the prefetcher —
        # so a mid-epoch resume never fetches features for a batch it is about to discard.
        iterator = iter(batches)
        for _ in range(resume_offset):
            next(iterator, None)
        with PrefetchingFeatureLoader(
            iterator, self.feature_provider, device=self.device, non_blocking=non_blocking
        ) as prefetcher:
            display_batches = progress_iter(
                prefetcher,
                desc=f"epoch {self.state.epoch}",
                unit="batch",
                disable=not (self.show_progress and self.distributed.is_main_process),
            )
            for offset, (_, batch_features, targets) in enumerate(display_batches):
                batch_index = resume_offset + offset
                with self.precision.autocast():
                    features = batch_features
                    if self.feature_augmentation is not None:
                        features = self.feature_augmentation.apply(features)
                    outputs = self.task.forward(features)
                    loss = self.task.compute_loss(outputs, targets)
                    scaled_loss = loss / self.config.accumulation_steps

                if train_metric is not None:
                    with torch.no_grad():
                        train_metric.update(outputs, targets)

                self.precision.backward(scaled_loss)
                self.state.micro_step += 1
                self.state.batch_in_epoch = batch_index + 1
                accumulated_steps += 1
                last_loss_value = loss.item()

                if accumulated_steps == self.config.accumulation_steps:
                    self._optimizer_step(last_loss_value)
                    accumulated_steps = 0

        if accumulated_steps:
            gradient_scale = self.config.accumulation_steps / accumulated_steps
            self._optimizer_step(last_loss_value, gradient_scale=gradient_scale)

    def _after_train_epoch(self, train_metric: Metric) -> None:
        metrics = train_metric.compute()
        self.last_train_metrics = metrics
        if self.distributed.is_main_process:
            logger.info("epoch %d train metrics: %s", self.state.epoch, metrics)
            self._log_metrics(
                {f"train_{name}": value for name, value in metrics.items()}, step=self.state.global_optimizer_step
            )

    def _optimizer_step(self, last_loss_value: float, *, gradient_scale: float = 1.0) -> None:
        if self.config.grad_clip_norm is not None:
            self.precision.unscale_(self.optimizer)
            self._scale_gradients(gradient_scale)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.config.grad_clip_norm)
        else:
            self._scale_gradients(gradient_scale)

        self.precision.step(self.optimizer)
        self.optimizer.zero_grad()
        scheduler = self.scheduler
        if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step()
        self.state.global_optimizer_step += 1

        if self.distributed.is_main_process:
            self._log_metrics(
                {"train_loss": last_loss_value, "lr": self.optimizer.param_groups[0]["lr"]},
                step=self.state.global_optimizer_step,
            )

        if (
            self.config.checkpoint_every_n_steps is not None
            and self.state.global_optimizer_step % self.config.checkpoint_every_n_steps == 0
        ):
            self._checkpoint(periodic=True, is_best=False, metric_value=None)

    def _scale_gradients(self, scale: float) -> None:
        if scale == 1.0:
            return
        for parameter in self.decoder.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)

    def _after_validation(
        self, metrics: dict[str, float], metric_name: str, higher_is_better: bool, *, next_epoch: int
    ) -> None:
        """Record validation results, then advance ``self.state`` to ``next_epoch`` before checkpointing.

        The state must advance *before* the checkpoint write below, not after (as ``fit()``'s own
        no-validation branch does): otherwise ``last.ckpt``/the periodic file would briefly pair a
        stale epoch number with ``batch_in_epoch`` already equal to a full epoch's length, and a
        crash in that window would make a resumed run silently replay a no-op epoch (see
        ``_train_one_epoch``'s resume-skip logic). Logging above still reports the just-completed
        epoch's number, since it reads ``self.state.epoch`` before this advances it.
        """
        value = metrics[metric_name]
        is_best = self._is_new_best(value, higher_is_better)
        if is_best:
            self.state.best_metric = value
            self.state.best_metric_name = metric_name
            self.state.epochs_without_improvement = 0
        else:
            self.state.epochs_without_improvement += 1

        scheduler = self.scheduler
        if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
            # ReduceLROnPlateau is always built with mode="min" (schedulers.py) — negate a
            # higher-is-better metric so "improvement" means the same thing to the scheduler.
            scheduler.step(-value if higher_is_better else value)

        if self.distributed.is_main_process:
            logger.info("epoch %d val metrics: %s (best=%s)", self.state.epoch, metrics, is_best)
            self._log_metrics(
                {f"val_{name}": val for name, val in metrics.items()}, step=self.state.global_optimizer_step
            )

        # CheckpointManager's top-k pruning ranks "lower is better"; negate a higher-is-better
        # metric so the best-performing validated epochs are the ones top-k retains.
        ranking_value = -value if higher_is_better else value
        self.state.epoch = next_epoch
        self.state.batch_in_epoch = 0
        self._checkpoint(periodic=True, is_best=is_best, metric_value=ranking_value)

    def _log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        if self.tracker is None:
            return
        try:
            self.tracker.log_metrics(metrics, step=step)
        except Exception:
            logger.exception("tracking backend failed while logging metrics; training will continue")

    def _is_new_best(self, value: float, higher_is_better: bool) -> bool:
        if self.state.best_metric is None:
            return True
        return value > self.state.best_metric if higher_is_better else value < self.state.best_metric

    def _should_stop_early(self) -> bool:
        patience = self.config.early_stopping_patience
        return patience is not None and self.state.epochs_without_improvement >= patience

    def _checkpoint(self, *, periodic: bool, is_best: bool, metric_value: float | None) -> None:
        if self.checkpoint_manager is None or not self.distributed.is_main_process:
            return
        checkpoint = Checkpoint(
            decoder_state=self.decoder.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict() if self.scheduler is not None else {},
            scaler_state=self.precision.state_dict(),
            training_state=self.state,
            rng_state=capture_rng_state(),
            fingerprints=self.config.fingerprints,
        )
        self.checkpoint_manager.save(checkpoint, is_best=is_best, periodic=periodic, metric_value=metric_value)

    def _resume(self) -> None:
        if self.checkpoint_manager is None:  # pragma: no cover — only called when it is set
            raise RuntimeError("_resume called without a checkpoint_manager")
        checkpoint = self.checkpoint_manager.load(
            self.checkpoint_manager.last_path, expected_fingerprints=self.config.fingerprints, strict=True
        )
        self.decoder.load_state_dict(checkpoint.decoder_state)
        self.optimizer.load_state_dict(checkpoint.optimizer_state)
        if self.scheduler is not None and checkpoint.scheduler_state:
            self.scheduler.load_state_dict(checkpoint.scheduler_state)
        self.precision.load_state_dict(checkpoint.scaler_state)
        self.state = checkpoint.training_state
        restore_rng_state(checkpoint.rng_state)
        logger.info(
            "resumed from checkpoint at epoch %d, optimizer step %d",
            self.state.epoch,
            self.state.global_optimizer_step,
        )

    def _device(self) -> torch.device:
        try:
            return next(self.decoder.parameters()).device
        except StopIteration:
            return torch.device("cpu")
