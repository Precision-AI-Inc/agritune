# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Server-wide configuration for the FastAPI layer."""

import os
from pathlib import Path

API_ROOT_VARIABLE = "AGRITUNE_API_ROOT"


def get_api_root() -> Path:
    """Return the directory every request-supplied path must resolve within.

    Configured via the ``AGRITUNE_API_ROOT`` environment variable; defaults to the current
    working directory the server process was started from when unset. Read fresh on every call
    so tests (and deployments) can point it at a different directory without restarting the
    process.

    Returns
    -------
    Path
    """
    return Path(os.environ.get(API_ROOT_VARIABLE, ".")).resolve()
