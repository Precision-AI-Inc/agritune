# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.tracking.comet_tracker.CometTracker.

``comet_ml`` is an optional extra (``pai-agritune[tracking]``) and is not installed in the default
test environment — this only exercises the missing-dependency error path, matching CLAUDE.md's
optional-dependency pattern.
"""

import pytest

from precisionai.agritune.tracking.comet_tracker import CometTracker


def test_raises_import_error_with_install_hint_when_comet_missing() -> None:
    with pytest.raises(ImportError, match=r"pip install pai-agritune\[tracking\]"):
        CometTracker(project_name="agritune-test")
