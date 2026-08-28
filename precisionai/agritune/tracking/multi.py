# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""``MultiTracker`` — fans every call out to a list of trackers, e.g. JSONL + TensorBoard at once."""

from typing import Any

from precisionai.agritune.schemas.protocols import Tracker


def _close_quietly(tracker: Tracker) -> Exception | None:
    try:
        tracker.close()
    except Exception as exc:  # intentionally broad: must not skip remaining closes
        return exc
    return None


class MultiTracker:
    """Broadcasts every call to each of ``trackers``, in order.

    Parameters
    ----------
    trackers : list[Tracker]
        Trackers to fan out to.
    """

    def __init__(self, trackers: list[Tracker]) -> None:
        self._trackers = trackers

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        """Forward to every tracker's ``log_metrics``."""
        for tracker in self._trackers:
            tracker.log_metrics(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        """Forward to every tracker's ``log_params``."""
        for tracker in self._trackers:
            tracker.log_params(params)

    def log_artifact(self, path: str) -> None:
        """Forward to every tracker's ``log_artifact``."""
        for tracker in self._trackers:
            tracker.log_artifact(path)

    def close(self) -> None:
        """Close every tracker, even if an earlier one raises.

        Raises
        ------
        Exception
            The first exception raised while closing, if any; every tracker is still given a
            chance to close before it propagates.
        """
        errors = [error for tracker in self._trackers if (error := _close_quietly(tracker)) is not None]
        if errors:
            raise errors[0]
