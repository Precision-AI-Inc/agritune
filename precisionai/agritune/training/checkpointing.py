# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Checkpointing: a checkpoint holds far more than model weights.

Decoder, optimizer, and scheduler state; the gradient scaler; full training progress and RNG
state; and fingerprints of the configuration, dataset, and encoder used to produce it. Resuming
warns or fails when critical fingerprints differ (encoder revision changed, class count changed,
decoder architecture changed).
"""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import torch

from precisionai.agritune.logging import get_logger
from precisionai.agritune.training.state import TrainingState

logger = get_logger(__name__)


class CheckpointMismatchError(Exception):
    """Raised when a checkpoint's fingerprints don't match the current run's, and ``strict=True``."""


@dataclass
class Checkpoint:
    """Everything needed to resume training exactly.

    Attributes
    ----------
    decoder_state : dict[str, Any]
        ``decoder.state_dict()``.
    optimizer_state : dict[str, Any]
        ``optimizer.state_dict()``.
    scheduler_state : dict[str, Any]
        ``scheduler.state_dict()``.
    scaler_state : dict[str, Any]
        ``PrecisionContext.state_dict()`` (the gradient scaler).
    training_state : TrainingState
        Epoch, step counters, and best-metric tracking.
    rng_state : dict[str, Any]
        Output of :func:`~precisionai.agritune.training.state.capture_rng_state`.
    fingerprints : dict[str, str]
        Identifies what produced this checkpoint — typically ``"config"``, ``"dataset"``,
        ``"encoder"`` keys mapping to opaque fingerprint strings.
    """

    decoder_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any]
    scaler_state: dict[str, Any]
    training_state: TrainingState
    rng_state: dict[str, Any]
    fingerprints: dict[str, str] = field(default_factory=dict)


class CheckpointManager:
    """Writes ``last.ckpt``/``best.ckpt``/periodic checkpoints and loads them back.

    Top-``k`` periodic-checkpoint pruning is tracked in-memory for the life of this instance only
    — it is a best-effort disk-space optimization, not required for exact resume (which relies on
    ``last.ckpt``).

    Parameters
    ----------
    directory : str | Path
        Directory to write checkpoints into; created if missing.
    top_k : int, optional
        Maximum number of periodic checkpoints to retain, ranked by metric value (lower is
        better). Set to ``0`` to keep every periodic checkpoint.
    """

    def __init__(self, directory: str | Path, *, top_k: int = 3) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._top_k = top_k
        self._ranked: list[tuple[float, Path]] = []

    @property
    def last_path(self) -> Path:
        """Path ``last.ckpt`` is (or will be) written to."""
        return self._directory / "last.ckpt"

    @property
    def best_path(self) -> Path:
        """Path ``best.ckpt`` is (or will be) written to."""
        return self._directory / "best.ckpt"

    def has_last(self) -> bool:
        """Return whether a ``last.ckpt`` exists to resume from."""
        return self.last_path.is_file()

    def save(
        self,
        checkpoint: Checkpoint,
        *,
        is_best: bool = False,
        periodic: bool = False,
        metric_value: float | None = None,
    ) -> Path:
        """Write ``checkpoint`` to ``last.ckpt``, and optionally ``best.ckpt``/a periodic file.

        Parameters
        ----------
        checkpoint : Checkpoint
            The checkpoint to persist.
        is_best : bool, optional
            Also write ``best.ckpt``.
        periodic : bool, optional
            Also write a step-numbered periodic checkpoint file.
        metric_value : float | None, optional
            Used to rank periodic checkpoints for top-``k`` pruning (lower is better); ignored
            unless ``periodic=True``.

        Returns
        -------
        Path
            The path ``last.ckpt`` was written to.
        """
        payload = self._to_payload(checkpoint)
        torch.save(payload, self.last_path)

        if is_best:
            torch.save(payload, self.best_path)

        if periodic:
            step_path = self._directory / f"step_{checkpoint.training_state.global_optimizer_step:08d}.ckpt"
            torch.save(payload, step_path)
            if metric_value is not None:
                self._prune_to_top_k(step_path, metric_value)

        return self.last_path

    def load(
        self,
        path: str | Path,
        *,
        expected_fingerprints: dict[str, str] | None = None,
        strict: bool = True,
    ) -> Checkpoint:
        """Load a checkpoint, optionally validating it against the current run's fingerprints.

        Parameters
        ----------
        path : str | Path
            Checkpoint file to load (e.g. :attr:`last_path`).
        expected_fingerprints : dict[str, str] | None, optional
            Fingerprints the current run expects (e.g. ``{"encoder": "..."}``); any mismatch
            against what's stored in the checkpoint is reported.
        strict : bool, optional
            Raise on mismatch (default) rather than only logging a warning.

        Returns
        -------
        Checkpoint

        Raises
        ------
        CheckpointMismatchError
            If ``strict=True`` and any expected fingerprint doesn't match the checkpoint's.
        """
        payload = torch.load(path, map_location="cpu", weights_only=False)
        fingerprints = payload["fingerprints"]

        if expected_fingerprints:
            self._check_fingerprints(fingerprints, expected_fingerprints, strict=strict)

        allowed = {item.name for item in fields(TrainingState)}
        training_payload = {key: value for key, value in payload["training_state"].items() if key in allowed}
        return Checkpoint(
            decoder_state=payload["decoder_state"],
            optimizer_state=payload["optimizer_state"],
            scheduler_state=payload["scheduler_state"],
            scaler_state=payload["scaler_state"],
            training_state=TrainingState(**training_payload),
            rng_state=payload["rng_state"],
            fingerprints=fingerprints,
        )

    @staticmethod
    def _to_payload(checkpoint: Checkpoint) -> dict[str, Any]:
        return {
            "decoder_state": checkpoint.decoder_state,
            "optimizer_state": checkpoint.optimizer_state,
            "scheduler_state": checkpoint.scheduler_state,
            "scaler_state": checkpoint.scaler_state,
            "training_state": asdict(checkpoint.training_state),
            "rng_state": checkpoint.rng_state,
            "fingerprints": checkpoint.fingerprints,
        }

    @staticmethod
    def _check_fingerprints(actual: dict[str, str], expected: dict[str, str], *, strict: bool) -> None:
        mismatches = {
            key: {"checkpoint": actual.get(key), "current": value}
            for key, value in expected.items()
            if actual.get(key) != value
        }
        if not mismatches:
            return
        message = f"checkpoint fingerprint mismatch: {mismatches}"
        if strict:
            raise CheckpointMismatchError(message)
        logger.warning(message)

    def _prune_to_top_k(self, path: Path, metric_value: float) -> None:
        if self._top_k <= 0:
            return
        self._ranked.append((metric_value, path))
        self._ranked.sort(key=lambda entry: entry[0])
        while len(self._ranked) > self._top_k:
            _, stale_path = self._ranked.pop()
            if stale_path.is_file():
                stale_path.unlink()
