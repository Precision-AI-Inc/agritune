# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Populate ``os.environ`` from a ``.env`` file before configuration is resolved.

OmegaConf's ``oc.env`` resolver reads ``os.environ`` and nothing else — neither OmegaConf 2.3 nor
Hydra 1.3 ships any ``.env`` support. Loading a ``.env`` file into the environment *before* config
composition therefore lets an interpolation like ``${oc.env:AGRITUNE_ENCODER_API_KEY,null}``
resolve from that file with no change to the config itself.

Shell-exported variables always take precedence: a ``.env`` file never overwrites a variable that
is already set, so an explicitly exported key still wins over the file.
"""

import os
from pathlib import Path

from precisionai.agritune.logging import get_logger

try:
    import dotenv as _dotenv

    _DOTENV_AVAILABLE = True
except ImportError:
    _dotenv = None
    _DOTENV_AVAILABLE = False

logger = get_logger(__name__)

ENV_FILE_VARIABLE = "AGRITUNE_ENV_FILE"
ENCODER_API_KEY_VARIABLE = "AGRITUNE_ENCODER_API_KEY"

_INSTALL_HINT = "python-dotenv is required to read a .env file: pip install pai-agritune[dotenv]"


def load_env_file(path: str | Path | None = None) -> Path | None:
    """Load environment variables from a ``.env`` file into ``os.environ``.

    Variables already present in the environment are never overwritten, so an exported value
    always beats the file. Safe to call more than once.

    Parameters
    ----------
    path : str | Path | None, optional
        Path to the ``.env`` file. When ``None``, the ``AGRITUNE_ENV_FILE`` environment variable is
        used if set; otherwise the nearest ``.env`` file is searched for from the working directory
        upwards.

    Returns
    -------
    Path | None
        The file whose variables were loaded, or ``None`` when no ``.env`` file was found (or one
        was discovered but ``python-dotenv`` is not installed).

    Raises
    ------
    FileNotFoundError
        If ``path`` — or ``AGRITUNE_ENV_FILE`` — names a file that does not exist.
    ImportError
        If a file is requested explicitly but ``python-dotenv`` is not installed.
    """
    requested = path if path is not None else os.environ.get(ENV_FILE_VARIABLE)
    if requested is not None:
        env_file = Path(requested)
        if not env_file.is_file():
            raise FileNotFoundError(f"env file not found: {env_file}")
        if not _DOTENV_AVAILABLE:
            raise ImportError(_INSTALL_HINT)
        return _load(env_file)

    discovered = _discover_env_file()
    if discovered is None:
        return None
    if not _DOTENV_AVAILABLE:
        # Discovery is implicit, so a missing optional dependency must not be fatal here — but it
        # must not be silent either, or the key the user put in .env goes missing without a word.
        logger.warning("Found %s but ignoring it: %s", discovered, _INSTALL_HINT)
        return None
    return _load(discovered)


def _discover_env_file() -> Path | None:
    """Return the nearest ``.env`` file at or above the working directory, if any."""
    cwd = Path.cwd().resolve()
    for directory in (cwd, *cwd.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def _load(env_file: Path) -> Path:
    if _dotenv is None:  # pragma: no cover - guarded by load_env_file
        raise ImportError(_INSTALL_HINT) from None
    _dotenv.load_dotenv(dotenv_path=env_file, override=False)
    # Only the path is ever logged: a .env file holds secrets, so its contents must not be.
    logger.debug("Loaded environment variables from %s", env_file)
    return env_file
