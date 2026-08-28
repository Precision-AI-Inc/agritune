# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""``MLflowTracker`` — an optional MLflow tracking backend.

Requires the ``mlflow`` package (``pip install pai-agritune[tracking]``); never required to run
AgriTune otherwise.
"""

from typing import Any

try:
    import mlflow

    _MLFLOW_AVAILABLE = True
except ImportError:
    mlflow: Any = None
    _MLFLOW_AVAILABLE = False


class MLflowTracker:
    """Logs metrics/params/artifacts to an MLflow run.

    Parameters
    ----------
    experiment_name : str | None, optional
        MLflow experiment to log under; ``None`` uses MLflow's default experiment.
    tracking_uri : str | None, optional
        MLflow tracking server URI; ``None`` uses MLflow's configured default (typically a local
        ``mlruns/`` directory).
    run_name : str | None, optional
        Name for the created MLflow run.

    Raises
    ------
    ImportError
        If the ``mlflow`` package is not installed.
    """

    def __init__(
        self,
        *,
        experiment_name: str | None = None,
        tracking_uri: str | None = None,
        run_name: str | None = None,
    ) -> None:
        if not _MLFLOW_AVAILABLE:
            raise ImportError("mlflow is required for MLflowTracker: pip install pai-agritune[tracking]") from None
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
        if experiment_name is not None:
            mlflow.set_experiment(experiment_name)
        self._run = mlflow.start_run(run_name=run_name)

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        """Log each metric to the active MLflow run at ``step``."""
        mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log ``params`` to the active MLflow run."""
        mlflow.log_params(params)

    def log_artifact(self, path: str) -> None:
        """Upload the file at ``path`` as an MLflow artifact."""
        mlflow.log_artifact(path)

    def close(self) -> None:
        """End the active MLflow run."""
        mlflow.end_run()
