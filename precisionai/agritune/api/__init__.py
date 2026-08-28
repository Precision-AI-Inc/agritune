# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0
"""Thin FastAPI layer over ``precisionai.agritune.services`` — no business logic lives here."""

from precisionai.agritune.api.app import create_app

__all__ = ["create_app"]
