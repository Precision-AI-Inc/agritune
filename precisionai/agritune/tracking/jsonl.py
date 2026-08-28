# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""``JSONLTracker`` — appends one JSON object per line; the always-available local tracker."""

import json
import time
from pathlib import Path
from typing import Any


class JSONLTracker:
    """Writes metrics/params/artifacts as JSON Lines to a file (typically ``runs/<id>/metrics.jsonl``).

    Parameters
    ----------
    path : str | Path
        File to append to; parent directory created if missing. Opened in append mode so a
        resumed run's history is preserved.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a", encoding="utf-8")

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        """Append one line: ``{"type": "metrics", "step": ..., "timestamp": ..., "data": ...}``."""
        self._write({"type": "metrics", "step": step, "timestamp": time.time(), "data": metrics})

    def log_params(self, params: dict[str, Any]) -> None:
        """Append one line: ``{"type": "params", "timestamp": ..., "data": ...}``."""
        self._write({"type": "params", "timestamp": time.time(), "data": params})

    def log_artifact(self, path: str) -> None:
        """Append one line: ``{"type": "artifact", "timestamp": ..., "path": ...}``."""
        self._write({"type": "artifact", "timestamp": time.time(), "path": path})

    def close(self) -> None:
        """Flush and close the underlying file."""
        self._file.close()

    def _write(self, record: dict[str, Any]) -> None:
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()
