# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.tracking.null.NullTracker."""

from precisionai.agritune.schemas.protocols import Tracker
from precisionai.agritune.tracking.null import NullTracker


def test_null_tracker_accepts_all_calls_without_error() -> None:
    tracker = NullTracker()
    tracker.log_metrics({"loss": 1.0}, step=0)
    tracker.log_params({"lr": 0.1})
    tracker.log_artifact("checkpoint.pt")
    tracker.close()


def test_null_tracker_satisfies_protocol() -> None:
    assert isinstance(NullTracker(), Tracker)
