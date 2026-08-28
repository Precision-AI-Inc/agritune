# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0
"""The ``agritune`` command-line entry point, re-exported for convenient import."""

from precisionai.agritune.cli.main import build_parser, main

__all__ = [
    "build_parser",
    "main",
]
