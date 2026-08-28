# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""``WandBTracker`` — an optional Weights & Biases tracking backend.

Requires the ``wandb`` package (``pip install pai-agritune[tracking]``); never required to run
AgriTune otherwise.
"""

from typing import Any

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    wandb: Any = None
    _WANDB_AVAILABLE = False


class WandBTracker:
    """Logs metrics/params/artifacts to a Weights & Biases run.

    Parameters
    ----------
    project : str
        W&B project name.
    run_name : str | None, optional
        Name for the created W&B run.

    Raises
    ------
    ImportError
        If the ``wandb`` package is not installed.
    """

    def __init__(self, *, project: str, run_name: str | None = None) -> None:
        if not _WANDB_AVAILABLE:
            raise ImportError("wandb is required for WandBTracker: pip install pai-agritune[tracking]") from None
        self._run = wandb.init(project=project, name=run_name)

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        """Log each metric to the active W&B run at ``step``."""
        self._run.log(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        """Update the active W&B run's config with ``params``."""
        self._run.config.update(params)

    def log_artifact(self, path: str) -> None:
        """Upload the file at ``path`` as a W&B artifact."""
        self._run.log_artifact(path)

    def close(self) -> None:
        """Finish the active W&B run."""
        self._run.finish()
