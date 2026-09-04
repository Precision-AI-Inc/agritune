# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Generates a fully-commented, flat training-config template for ``agritune config init``.

The packaged template at ``precisionai/agritune/configs/templates/full_config.yaml`` spells out
every field :class:`~precisionai.agritune.services.training_service.TrainingRunConfig` accepts,
each already set to its default value, with dataset-specific fields marked ``REQUIRED`` as
placeholders — see ``examples/segmentation/cwfid.yaml`` for a filled-in, working example of the
same shape.
"""

from pathlib import Path

from precisionai.agritune.logging import get_logger
from precisionai.agritune.utils.scaffold import read_scaffold_template, write_scaffold_file

logger = get_logger(__name__)


def render_config_template() -> str:
    """Return the packaged full training-config template's text.

    Returns
    -------
    str
    """
    return read_scaffold_template("full_config.yaml")


def write_config_template(output_path: str | Path, *, force: bool = False) -> Path:
    """Write the full training-config template to ``output_path``.

    Parameters
    ----------
    output_path : str | Path
        Destination file path. Parent directories are created as needed.
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
    path = write_scaffold_file(output_path, render_config_template(), force=force)
    logger.info("wrote config template to %s", path)
    return path
