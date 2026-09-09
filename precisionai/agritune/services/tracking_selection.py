# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tracking backend selection shared by the CLI, API layer, and Hydra ``tracking`` config group.

Turns a plain ``backends`` list (``["jsonl", "tensorboard"]``, matching the Hydra
``tracking.backends`` config field) plus a handful of
per-backend settings into a single :class:`~precisionai.agritune.schemas.protocols.Tracker` —
``NullTracker`` if empty, the one tracker directly if there is exactly one, or a ``MultiTracker``
fanning out to all of them.
"""

from dataclasses import dataclass, field

from precisionai.agritune.logging import RunDirectory, get_logger
from precisionai.agritune.schemas.protocols import Tracker
from precisionai.agritune.tracking.comet_tracker import CometTracker
from precisionai.agritune.tracking.jsonl import JSONLTracker
from precisionai.agritune.tracking.mlflow_tracker import MLflowTracker
from precisionai.agritune.tracking.multi import MultiTracker
from precisionai.agritune.tracking.neptune_tracker import NeptuneTracker
from precisionai.agritune.tracking.null import NullTracker
from precisionai.agritune.tracking.tensorboard import TensorBoardTracker
from precisionai.agritune.tracking.wandb_tracker import WandBTracker

_KNOWN_BACKENDS = frozenset({"null", "jsonl", "tensorboard", "mlflow", "wandb", "neptune", "comet"})
logger = get_logger(__name__)


@dataclass
class TrackingSelection:
    """Which tracking backend(s) to use, and their per-backend settings.

    Attributes
    ----------
    backends : list[str]
        Any of ``"null"``, ``"jsonl"``, ``"tensorboard"``, ``"mlflow"``, ``"wandb"``,
        ``"neptune"``, ``"comet"``. An empty list is equivalent to ``["null"]``.
    mlflow_experiment_name : str | None
    mlflow_tracking_uri : str | None
    wandb_project : str
    neptune_project : str | None
        Required (in ``workspace/project`` form) if ``"neptune"`` is selected.
    comet_project_name : str
    """

    backends: list[str] = field(default_factory=lambda: ["jsonl"])
    mlflow_experiment_name: str | None = None
    mlflow_tracking_uri: str | None = None
    wandb_project: str = "agritune"
    neptune_project: str | None = None
    comet_project_name: str = "agritune"


def build_trackers(selection: TrackingSelection, *, run_dir: RunDirectory, run_id: str) -> Tracker:
    """Construct the ``Tracker`` (or fan-out of trackers) named in ``selection``.

    Parameters
    ----------
    selection : TrackingSelection
    run_dir : RunDirectory
        Supplies default paths for the local backends (JSONL, TensorBoard).
    run_id : str
        Used as the run/experiment name for external backends that need one.

    Returns
    -------
    Tracker

    Raises
    ------
    ValueError
        If ``selection.backends`` names an unknown backend, or ``"neptune"`` is selected without
        ``neptune_project``.
    """
    unknown = sorted(set(selection.backends) - _KNOWN_BACKENDS)
    if unknown:
        raise ValueError(f"unknown tracking backend(s): {unknown}; known backends: {sorted(_KNOWN_BACKENDS)}")

    backends = [name for name in selection.backends if name != "null"]
    if not backends:
        logger.info("using NullTracker (no tracking backend selected)")
        return NullTracker()

    logger.info("using tracking backend(s): %s", backends)
    trackers = [_build_one(name, selection, run_dir=run_dir, run_id=run_id) for name in backends]
    return trackers[0] if len(trackers) == 1 else MultiTracker(trackers)


def _build_one(name: str, selection: TrackingSelection, *, run_dir: RunDirectory, run_id: str) -> Tracker:
    if name == "jsonl":
        return JSONLTracker(run_dir.metrics_path)
    if name == "tensorboard":
        return TensorBoardTracker(run_dir.path / "tensorboard")
    if name == "mlflow":
        return MLflowTracker(
            experiment_name=selection.mlflow_experiment_name,
            tracking_uri=selection.mlflow_tracking_uri,
            run_name=run_id,
        )
    if name == "wandb":
        return WandBTracker(project=selection.wandb_project, run_name=run_id)
    if name == "neptune":
        if not selection.neptune_project:
            raise ValueError("neptune_project is required when 'neptune' is a selected tracking backend")
        return NeptuneTracker(project=selection.neptune_project, run_name=run_id)
    if name == "comet":
        return CometTracker(project_name=selection.comet_project_name)
    raise ValueError(f"unknown tracking backend: {name!r}")  # pragma: no cover — guarded by build_trackers
