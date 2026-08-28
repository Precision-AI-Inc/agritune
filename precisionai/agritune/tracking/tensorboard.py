# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""``TensorBoardTracker`` — an optional local tracking backend.

Requires the ``tensorboard`` package (``pip install pai-agritune[tracking]``); never required to
run AgriTune otherwise.
"""

from pathlib import Path
from typing import Any

try:
    from torch.utils.tensorboard import SummaryWriter

    _TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter: Any = None
    _TENSORBOARD_AVAILABLE = False


class TensorBoardTracker:
    """Writes scalars to a TensorBoard event file under ``log_dir``.

    Parameters
    ----------
    log_dir : str | Path
        Directory TensorBoard event files are written into.

    Raises
    ------
    ImportError
        If the ``tensorboard`` package is not installed.
    """

    def __init__(self, log_dir: str | Path) -> None:
        if not _TENSORBOARD_AVAILABLE:
            raise ImportError(
                "tensorboard is required for TensorBoardTracker: pip install pai-agritune[tracking]"
            ) from None
        self._writer = SummaryWriter(log_dir=str(log_dir))

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        """Write each metric as a scalar at ``step``."""
        for name, value in metrics.items():
            self._writer.add_scalar(name, value, global_step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        """Write ``params`` as a text summary (TensorBoard has no native hyperparameter dict)."""
        self._writer.add_text("params", str(params))

    def log_artifact(self, path: str) -> None:
        """Record an artifact path as a text summary (TensorBoard does not store files itself)."""
        self._writer.add_text("artifact", path)

    def close(self) -> None:
        """Flush and close the event file writer."""
        self._writer.close()
