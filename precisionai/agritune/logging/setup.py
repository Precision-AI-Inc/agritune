# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Structured logging setup for AgriTune.

All AgriTune loggers live under the ``"agritune"`` logger namespace. Call
:func:`configure_logging` once, early in the CLI/API entry point, before any other AgriTune code
logs a message.
"""

import logging

from rich.logging import RichHandler

_LOGGER_NAME = "agritune"


def configure_logging(level: str = "INFO") -> None:
    """Configure the root ``agritune`` logger with a Rich console handler.

    Idempotent: calling this more than once replaces the previous handlers rather than
    accumulating duplicate log lines.

    Parameters
    ----------
    level : str, optional
        Logging level name (e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``). Case-insensitive.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.handlers.clear()
    handler = RichHandler(show_path=False, rich_tracebacks=True)
    handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger nested under the ``"agritune"`` namespace.

    Parameters
    ----------
    name : str
        Suffix appended to the ``"agritune"`` logger name, typically ``__name__`` of the caller.

    Returns
    -------
    logging.Logger
        A logger named ``f"agritune.{name}"``.
    """
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")
