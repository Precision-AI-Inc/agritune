# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.tracking.tensorboard.TensorBoardTracker.

``tensorboard`` is an optional extra (``pai-agritune[tracking]``) — these tests are skipped
entirely when it isn't installed, rather than asserting real writer behavior against a stub.
"""

from pathlib import Path

import pytest

pytest.importorskip("tensorboard")

from precisionai.agritune.schemas.protocols import Tracker
from precisionai.agritune.tracking.tensorboard import TensorBoardTracker


def test_logs_and_writes_an_event_file(tmp_path: Path) -> None:
    tracker = TensorBoardTracker(tmp_path)
    tracker.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=1)
    tracker.log_params({"lr": 0.1})
    tracker.log_artifact("checkpoints/best.ckpt")
    tracker.close()

    event_files = list(tmp_path.glob("events.out.tfevents.*"))
    assert event_files


def test_tensorboard_tracker_satisfies_protocol(tmp_path: Path) -> None:
    tracker = TensorBoardTracker(tmp_path)
    assert isinstance(tracker, Tracker)
    tracker.close()
