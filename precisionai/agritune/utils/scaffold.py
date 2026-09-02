# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the ``agritune config init``/``agritune dataset init`` scaffold templates."""

from importlib import resources
from pathlib import Path

_TEMPLATE_ANCHOR = "precisionai.agritune"
_TEMPLATE_ROOT = ("configs", "templates")


def read_scaffold_template(filename: str) -> str:
    """Return the text of a packaged template under ``precisionai/agritune/configs/templates/``.

    Parameters
    ----------
    filename : str
        Template file name (e.g. ``"full_config.yaml"``, ``"manifest.csv"``).

    Returns
    -------
    str
    """
    template = resources.files(_TEMPLATE_ANCHOR)
    for part in (*_TEMPLATE_ROOT, filename):
        template = template.joinpath(part)
    return template.read_text(encoding="utf-8")


def write_scaffold_file(output_path: str | Path, content: str, *, force: bool = False) -> Path:
    """Write ``content`` to ``output_path``, creating parent directories as needed.

    Parameters
    ----------
    output_path : str | Path
        Destination file path.
    content : str
        Text to write.
    force : bool, optional
        Overwrite ``output_path`` if it already exists (default ``False``).

    Returns
    -------
    Path
        ``output_path``.

    Raises
    ------
    FileExistsError
        If ``output_path`` already exists and ``force`` is ``False``.
    """
    path = Path(output_path)
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
