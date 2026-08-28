# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""``CometTracker`` — an optional Comet ML tracking backend.

Requires the ``comet_ml`` package (``pip install pai-agritune[tracking]``); never required to run
AgriTune otherwise. Comet asks that ``comet_ml`` be imported before ``torch`` in the process that
uses it — see Comet's own setup docs; this module only performs that import when constructed.
"""

from typing import Any

try:
    import comet_ml

    _COMET_AVAILABLE = True
except ImportError:
    comet_ml: Any = None
    _COMET_AVAILABLE = False


class CometTracker:
    """Logs metrics/params/artifacts to a Comet ML experiment.

    Parameters
    ----------
    project_name : str
        Comet project name.
    api_key : str | None, optional
        Comet API key; ``None`` uses the ``COMET_API_KEY`` environment variable / Comet config.
    workspace : str | None, optional
        Comet workspace name; ``None`` uses the account's default workspace.

    Raises
    ------
    ImportError
        If the ``comet_ml`` package is not installed.
    """

    def __init__(self, *, project_name: str, api_key: str | None = None, workspace: str | None = None) -> None:
        if not _COMET_AVAILABLE:
            raise ImportError("comet_ml is required for CometTracker: pip install pai-agritune[tracking]") from None
        self._experiment = comet_ml.Experiment(project_name=project_name, api_key=api_key, workspace=workspace)

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        """Log each metric to the active Comet experiment at ``step``."""
        self._experiment.log_metrics(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log ``params`` to the active Comet experiment."""
        self._experiment.log_parameters(params)

    def log_artifact(self, path: str) -> None:
        """Upload the file at ``path`` as a Comet asset."""
        self._experiment.log_asset(path)

    def close(self) -> None:
        """End the active Comet experiment."""
        self._experiment.end()
