# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""``NeptuneTracker`` — an optional neptune.ai tracking backend.

Requires the ``neptune`` package (``pip install pai-agritune[tracking]``); never required to run
AgriTune otherwise.
"""

from pathlib import Path
from typing import Any

try:
    import neptune

    _NEPTUNE_AVAILABLE = True
except ImportError:
    neptune: Any = None
    _NEPTUNE_AVAILABLE = False


class NeptuneTracker:
    """Logs metrics/params/artifacts to a neptune.ai run.

    Parameters
    ----------
    project : str
        Neptune project name, in ``workspace/project`` form.
    api_token : str | None, optional
        Neptune API token; ``None`` uses the ``NEPTUNE_API_TOKEN`` environment variable.
    run_name : str | None, optional
        Name for the created Neptune run.

    Raises
    ------
    ImportError
        If the ``neptune`` package is not installed.
    """

    def __init__(self, *, project: str, api_token: str | None = None, run_name: str | None = None) -> None:
        if not _NEPTUNE_AVAILABLE:
            raise ImportError("neptune is required for NeptuneTracker: pip install pai-agritune[tracking]") from None
        self._run = neptune.init_run(project=project, api_token=api_token, name=run_name)

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        """Append each metric to the active Neptune run at ``step``."""
        for name, value in metrics.items():
            self._run[f"metrics/{name}"].append(value, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        """Record ``params`` under the active Neptune run's ``parameters`` namespace."""
        self._run["parameters"] = params

    def log_artifact(self, path: str) -> None:
        """Upload the file at ``path`` as a Neptune artifact."""
        self._run[f"artifacts/{Path(path).name}"].upload(path)

    def close(self) -> None:
        """Stop the active Neptune run."""
        self._run.stop()
