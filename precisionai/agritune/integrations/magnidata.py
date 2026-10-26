# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Run AgriTune inside MagniData's MLflow queue and artifact contract."""

import csv
import heapq
import itertools
import tempfile
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn import functional

try:
    import mlflow
    from mlflow.entities import Metric

    _MLFLOW_AVAILABLE = True
except ImportError:
    mlflow: Any = None
    Metric: Any = None
    _MLFLOW_AVAILABLE = False

from precisionai.agritune.data.dataset import ManifestDataset
from precisionai.agritune.features.store import DirectoryFeatureStore, ShardedFeatureStore
from precisionai.agritune.services.segmentation_common import build_training_batches
from precisionai.agritune.services.training_service import (
    TrainingRunConfig,
    TrainingRunResult,
    TrainingRuntime,
    run_training,
)
from precisionai.agritune.tasks.segmentation.losses import segmentation_losses_per_sample

_ARTIFACT_ROOT = "magnidata"
_CANCEL_TAG = "magnidata.cancel_requested"
_ERROR_TAG = "magnidata.error"
_DEFAULT_PALETTE = ((0, 0, 0), (67, 160, 71), (255, 193, 7), (211, 47, 47))
# Keys the trainer reports on every optimizer step; they are averaged into one value per epoch.
_PER_STEP_KEYS = ("train_loss", "lr")
ScoreRow = dict[str, str | float]


@dataclass(frozen=True)
class MagniDataOptions:
    """Configure per-epoch output produced for MagniData.

    Attributes
    ----------
    score_every : int
        Score all samples every N epochs; epoch one and the last epoch are always scored.
    top_k_predictions : int
        Persist this many highest-loss prediction masks per scored epoch.
    palette : tuple[tuple[int, int, int], ...]
        Class-index to RGB mapping used for prediction PNGs.
    image_names : Mapping[str, str]
        MagniData image name per manifest ``sample_id``, for manifests whose sample IDs are not the
        image names MagniData joins on (e.g. one reused with an existing feature store). Samples
        missing from the mapping are named by their ``sample_id``.
    """

    score_every: int = 1
    top_k_predictions: int = 50
    palette: tuple[tuple[int, int, int], ...] = _DEFAULT_PALETTE
    image_names: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate scoring intervals and prediction counts."""
        if self.score_every < 1:
            raise ValueError("score_every must be positive")
        if self.top_k_predictions < 0:
            raise ValueError("top_k_predictions must be non-negative")


class MagniDataTracker:
    """Log AgriTune metrics into an existing MagniData MLflow run, one point per epoch.

    The trainer reports the loss and learning rate on every optimizer step and the epoch metrics keyed by the global
    step. Writing those straight to MLflow would store thousands of points per run and put them on a different axis
    from the epoch-based losses MagniData expects, so values are held in memory and :meth:`flush_epoch` writes a
    single batched request per epoch, with the epoch number as the step.
    """

    def __init__(self, client: Any, run_id: str) -> None:
        self.client = client
        self.run_id = run_id
        self._step_values: dict[str, list[float]] = {}
        self._epoch_values: dict[str, float] = {}

    @staticmethod
    def _key(key: str) -> str:
        mappings = {
            "train_loss": "train_loss/loss",
            "lr": "Learning_rate/lr",
            "train_mean_iou": "train_metric/miou",
            "val_loss": "val_loss/loss",
            "val_mean_iou": "val_metric/miou",
        }
        if key in mappings:
            return mappings[key]
        if key.startswith("val_iou_class_"):
            return "val_metric/iou_" + key.removeprefix("val_iou_class_")
        if key.startswith("val_"):
            return "val_metric/" + key.removeprefix("val_")
        if key.startswith("train_"):
            return "train_metric/" + key.removeprefix("train_")
        return key

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        """Hold ``metrics`` until :meth:`flush_epoch`; ``step`` (the global optimizer step) is deliberately not used."""
        del step
        for key, value in metrics.items():
            if key in _PER_STEP_KEYS:
                self._step_values.setdefault(key, []).append(float(value))
            else:
                self._epoch_values[key] = float(value)

    def flush_epoch(self, epoch: int) -> None:
        """Log the held metrics as one point per key at ``epoch``: the mean per-step loss and the last learning rate."""
        values = dict(self._epoch_values)
        for key, series in self._step_values.items():
            values[key] = series[-1] if key == "lr" else sum(series) / len(series)
        self._step_values.clear()
        self._epoch_values.clear()
        if not values:
            return
        timestamp = int(time.time() * 1000)
        self.client.log_batch(
            self.run_id,
            metrics=[Metric(self._key(key), value, timestamp, epoch) for key, value in values.items()],
        )

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters which are not already present on the queued run."""
        existing = self.client.get_run(self.run_id).data.params
        for key, value in params.items():
            if key not in existing:
                self.client.log_param(self.run_id, key, value)

    def log_artifact(self, path: str) -> None:
        """Upload a file at the artifact root."""
        self.client.log_artifact(self.run_id, path)

    def close(self) -> None:
        """Leave terminal status ownership to :func:`run_magnidata_training`."""


def _split_map(runtime: TrainingRuntime) -> dict[str, str]:
    return {
        **dict.fromkeys(runtime.split.train, "train"),
        **dict.fromkeys(runtime.split.val, "val"),
        **dict.fromkeys(runtime.split.test, "test"),
    }


def _per_sample_iou(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int) -> torch.Tensor:
    if logits.shape[-2:] != targets.shape[-2:]:
        logits = functional.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
    predictions = logits.argmax(1)
    result = []
    for prediction, target in zip(predictions, targets, strict=True):
        valid = target != ignore_index
        intersection = ((prediction == target) & valid).sum()
        result.append(intersection.float() / valid.sum().clamp(min=1))
    return torch.stack(result)


def _confidence(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int) -> torch.Tensor:
    if logits.shape[-2:] != targets.shape[-2:]:
        logits = functional.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
    values = logits.softmax(1).amax(1)
    valid = targets != ignore_index
    return (values * valid).sum((1, 2)) / valid.sum((1, 2)).clamp(min=1)


class _HardestPredictions:
    """Keep the ``k`` highest-loss predicted masks seen so far, as ``uint8``.

    Scoring visits every sample but saves only the top ``k`` predictions; holding every mask until the end
    (int64, ~3.4 MB each at 798x532) needed hundreds of GB on large datasets.
    """

    def __init__(self, k: int) -> None:
        self.k = k
        self._heap: list[tuple[float, int, str, np.ndarray]] = []
        self._order = itertools.count()

    def __len__(self) -> int:
        return len(self._heap)

    def offer(self, loss: float, image_name: str, mask: np.ndarray) -> None:
        """Keep this prediction if it is among the ``k`` highest losses so far."""
        if self.k <= 0:
            return
        if len(self._heap) >= self.k and loss <= self._heap[0][0]:
            return
        item = (loss, next(self._order), image_name, mask.astype(np.uint8))
        if len(self._heap) < self.k:
            heapq.heappush(self._heap, item)
        else:
            heapq.heapreplace(self._heap, item)

    def ranked(self) -> list[tuple[float, str, np.ndarray]]:
        """Return the kept predictions, highest loss first."""
        return [(loss, name, mask) for loss, _, name, mask in sorted(self._heap, key=lambda item: (-item[0], item[1]))]


class _EpochPublisher:
    def __init__(
        self, client: Any, tracker: MagniDataTracker, run_id: str, runtime: TrainingRuntime, options: MagniDataOptions
    ) -> None:
        self.client = client
        self.tracker = tracker
        self.run_id = run_id
        self.runtime = runtime
        self.options = options
        self.splits = _split_map(runtime)
        self.scored_epochs: list[int] = []
        self.scratch = tempfile.TemporaryDirectory(prefix="agritune-magnidata-")
        self.root = Path(self.scratch.name)
        self._publish_split()

    def _image_name(self, sample_id: str) -> str:
        return self.options.image_names.get(sample_id, sample_id)

    def _tag(self, key: str, value: object) -> None:
        self.client.set_tag(self.run_id, key, str(value))

    def _publish_split(self) -> None:
        path = self.root / "split.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_name", "split"])
            writer.writerows((self._image_name(sample_id), split) for sample_id, split in self.splits.items())
        self.client.log_artifact(self.run_id, str(path), artifact_path=_ARTIFACT_ROOT)
        self._tag("magnidata.has_split", "true")

    def __call__(self, epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float]) -> bool:
        """Publish one completed epoch and return whether cancellation was requested."""
        del train_metrics, val_metrics
        self.tracker.flush_epoch(epoch)
        self._tag("magnidata.progress_stage", f"AgriTune epoch {epoch}/{self.runtime.config.trainer.max_epochs}")
        self._tag("magnidata.progress_current", epoch)
        self._tag("magnidata.progress_total", self.runtime.config.trainer.max_epochs)
        cancelled = self.client.get_run(self.run_id).data.tags.get(_CANCEL_TAG) == "true"
        final_epoch = epoch == self.runtime.config.trainer.max_epochs
        if not cancelled and (epoch == 1 or final_epoch or epoch % self.options.score_every == 0):
            self._score(epoch)
        self._upload_checkpoint()
        return cancelled

    def _score(self, epoch: int) -> None:
        self._tag("magnidata.progress_stage", f"AgriTune epoch {epoch} scoring")
        dataset = ManifestDataset(self.runtime.config.manifest_path)
        # Cached features never need the images; decoding them anyway dominated scoring time on large datasets.
        batches = build_training_batches(
            dataset,
            batch_size=self.runtime.config.batch_size,
            load_images=self.runtime.config.feature_provider != "cached",
        )
        rows: list[ScoreRow] = []
        hardest = _HardestPredictions(self.options.top_k_predictions)
        decoder = self.runtime.task.decoder
        decoder.eval()
        device = next(decoder.parameters()).device
        with torch.no_grad():
            for batch in batches:
                features = self.runtime.feature_provider.get_features(batch.samples).to(device)
                targets = batch.targets.to(device)
                logits = self.runtime.task.forward(features)
                losses = segmentation_losses_per_sample(
                    logits,
                    targets,
                    self.runtime.config.loss,
                    num_classes=self.runtime.config.num_classes,
                )
                ious = _per_sample_iou(logits, targets, self.runtime.config.loss.ignore_index)
                confidences = _confidence(logits, targets, self.runtime.config.loss.ignore_index)
                resized = logits
                if logits.shape[-2:] != targets.shape[-2:]:
                    resized = functional.interpolate(
                        logits, size=targets.shape[-2:], mode="bilinear", align_corners=False
                    )
                masks = resized.argmax(1).cpu().numpy()
                for index, sample in enumerate(batch.samples):
                    row: ScoreRow = {
                        "image_name": self._image_name(sample.sample_id),
                        "split": self.splits.get(sample.sample_id, "test"),
                        "loss_total": float(losses["total"][index]),
                        "iou": float(ious[index]),
                        "confidence": float(confidences[index]),
                    }
                    for term, values in losses.items():
                        if term != "total":
                            row[f"loss_{term}"] = float(values[index])
                    rows.append(row)
                    hardest.offer(float(losses["total"][index]), self._image_name(sample.sample_id), masks[index])
        self._write_losses(epoch, rows)
        self._write_predictions(epoch, hardest.ranked())
        self._log_loss_metrics(epoch, rows)
        self.scored_epochs.append(epoch)
        self._tag("magnidata.scored_epochs", ",".join(str(item) for item in self.scored_epochs))
        self._tag("magnidata.per_sample_schema", "magnidata-training/1.0.0")

    def _write_losses(self, epoch: int, rows: list[ScoreRow]) -> None:
        path = self.root / f"epoch_{epoch:04d}.csv"
        fieldnames = list(rows[0]) if rows else ["image_name", "split", "loss_total"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        self.client.log_artifact(self.run_id, str(path), artifact_path=f"{_ARTIFACT_ROOT}/per_sample")

    def _write_predictions(self, epoch: int, predictions: list[tuple[float, str, np.ndarray]]) -> None:
        palette = np.asarray(self.options.palette, dtype=np.uint8)
        for _, image_name, mask in predictions:
            rgb = palette[np.clip(mask, 0, len(palette) - 1)]
            path = self.root / f"{Path(image_name).stem}.png"
            Image.fromarray(rgb, mode="RGB").save(path)
            self.client.log_artifact(
                self.run_id, str(path), artifact_path=f"{_ARTIFACT_ROOT}/predictions/epoch_{epoch:04d}"
            )

    def _log_loss_metrics(self, epoch: int, rows: list[ScoreRow]) -> None:
        # The weighted totals are logged by the tracker (training mean and validation loss); only the components are added here.
        terms = (
            [key.removeprefix("loss_") for key in rows[0] if key.startswith("loss_") and key != "loss_total"]
            if rows
            else []
        )
        for split in ("train", "val"):
            selected = [row for row in rows if row["split"] == split]
            for term in terms:
                if selected:
                    value = sum(float(row[f"loss_{term}"]) for row in selected) / len(selected)
                    self.client.log_metric(self.run_id, f"{split}_loss/{term}", value, step=epoch)

    def _upload_checkpoint(self) -> None:
        last = self.runtime.run_directory.checkpoints_dir / "last.ckpt"
        if last.is_file():
            self.client.log_artifact(self.run_id, str(last), artifact_path="checkpoints")

    def close(self) -> None:
        """Release temporary scoring files."""
        self.scratch.cleanup()


def _describe_device(name: str) -> str:
    """Return ``cuda:0 (GPU name)`` for a CUDA device, else the device name, as MagniData shows it."""
    device = torch.device(name)
    if device.type != "cuda" or not torch.cuda.is_available():
        return str(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    return f"cuda:{index} ({torch.cuda.get_device_name(index)})"


def run_magnidata_training(
    config: TrainingRunConfig,
    *,
    store: DirectoryFeatureStore | ShardedFeatureStore,
    run_id: str,
    tracking_uri: str,
    options: MagniDataOptions | None = None,
    show_progress: bool = True,
) -> TrainingRunResult | None:
    """Train AgriTune into an existing queued MagniData run.

    The function owns the MLflow status transition and converts uncaught failures into a FAILED
    run with a short error tag. Cancellation is observed after each epoch and produces KILLED.
    """
    if not _MLFLOW_AVAILABLE:
        raise ImportError(
            "mlflow is required for MagniData integration: pip install 'pai-agritune[tracking]'"
        ) from None
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    client.update_run(run_id, status="RUNNING")
    client.set_tag(run_id, "magnidata.engine", "agritune")
    client.set_tag(run_id, "magnidata.device", _describe_device(config.device))
    tracker = MagniDataTracker(client, run_id)
    publisher: _EpochPublisher | None = None

    def factory(runtime: TrainingRuntime) -> _EpochPublisher:
        nonlocal publisher
        publisher = _EpochPublisher(client, tracker, run_id, runtime, options or MagniDataOptions())
        return publisher

    try:
        result = run_training(
            config,
            store=store,
            show_progress=show_progress,
            tracker_override=tracker,
            epoch_callback_factory=factory,
        )
        cancelled = client.get_run(run_id).data.tags.get(_CANCEL_TAG) == "true"
        status = "KILLED" if cancelled else "FINISHED"
        if not cancelled:
            for path in result.run_directory.path.glob("*.json"):
                client.log_artifact(run_id, str(path), artifact_path="agritune/provenance")
            for path in result.run_directory.path.glob("*.yaml"):
                client.log_artifact(run_id, str(path), artifact_path="agritune/provenance")
            client.set_tag(run_id, "magnidata.progress_stage", "Complete")
        client.set_terminated(run_id, status)
        return result
    except Exception as error:
        traceback.print_exc()
        client.set_tag(run_id, _ERROR_TAG, (str(error).splitlines() or [type(error).__name__])[0][:250])
        client.set_terminated(run_id, "FAILED")
        return None
    finally:
        if publisher is not None:
            publisher.close()
