# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""``NullTracker`` — the default tracking backend, which discards everything.

Nothing in AgriTune requires a tracking backend to run.
"""

from typing import Any


class NullTracker:
    """A :class:`~precisionai.agritune.schemas.protocols.Tracker` that does nothing."""

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        """Discard ``metrics``."""

    def log_params(self, params: dict[str, Any]) -> None:
        """Discard ``params``."""

    def log_artifact(self, path: str) -> None:
        """Discard the artifact reference at ``path``."""

    def close(self) -> None:
        """No resources to release."""
