# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.logging.setup."""

import logging

from precisionai.agritune.logging import configure_logging, get_logger


def test_get_logger_name_is_nested_under_agritune() -> None:
    logger = get_logger("mymodule")
    assert logger.name == "agritune.mymodule"


def test_configure_logging_sets_level() -> None:
    configure_logging("DEBUG")
    assert logging.getLogger("agritune").level == logging.DEBUG


def test_configure_logging_is_idempotent() -> None:
    configure_logging("INFO")
    configure_logging("INFO")
    assert len(logging.getLogger("agritune").handlers) == 1


def test_configure_logging_does_not_propagate_to_root() -> None:
    configure_logging("INFO")
    assert logging.getLogger("agritune").propagate is False
