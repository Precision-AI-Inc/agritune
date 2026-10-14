# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Dataset adapters producing :class:`Sample` objects from a manifest.

``ManifestDataset`` is the concrete adapter for the CSV manifest format
(``docs/datasets.md``); ``SegmentationDataset`` is the structural interface it (and any future
adapter) satisfies.
"""

from pathlib import Path
from typing import Protocol, runtime_checkable

from PIL import Image

from precisionai.agritune.data.manifest import ManifestRow, load_manifest
from precisionai.agritune.schemas.samples import Sample


@runtime_checkable
class SegmentationDataset(Protocol):
    """Structural interface for a segmentation dataset adapter."""

    def __len__(self) -> int:
        """Return the number of samples."""
        ...

    def __getitem__(self, index: int) -> Sample:
        """Return the sample at ``index``."""
        ...


class ManifestDataset:
    """A :class:`SegmentationDataset` backed by a CSV manifest.

    Parameters
    ----------
    manifest_path : str | Path
        Path to the manifest CSV (see ``precisionai.agritune.data.manifest``).
    sample_ids : list[str] | None, optional
        When given, restrict this dataset to only these sample IDs (e.g. one split's IDs from a
        :class:`~precisionai.agritune.data.split.SplitAssignment`). ``None`` keeps every row.
    """

    def __init__(self, manifest_path: str | Path, *, sample_ids: list[str] | None = None) -> None:
        self._base_dir = Path(manifest_path).parent
        rows = load_manifest(manifest_path)
        if sample_ids is not None:
            allowed = set(sample_ids)
            rows = [row for row in rows if row.sample_id in allowed]
        self._rows: list[ManifestRow] = rows

    def __len__(self) -> int:
        """Return the number of samples in this dataset."""
        return len(self._rows)

    def __getitem__(self, index: int) -> Sample:
        """Load and return the sample at ``index``.

        The image is loaded and converted to RGB; the mask is loaded in its native mode
        (typically single-channel integer labels) and left unconverted.
        """
        row = self._rows[index]
        image = Image.open(self._base_dir / row.image_path).convert("RGB")
        mask = Image.open(self._base_dir / row.mask_path)
        return Sample(sample_id=row.sample_id, image=image, target=mask, metadata=dict(row.metadata))

    def load_target(self, index: int) -> Sample:
        """Load and return the sample at ``index``, without ever opening its image (``image=None``).

        For ``feature_provider: cached`` batch-building, which never consults ``Sample.image`` (the
        encoder features are already computed) — skips the most expensive part of loading a sample
        entirely. Never valid for ``online``/``hybrid`` providers, which do need the real image.
        """
        row = self._rows[index]
        mask = Image.open(self._base_dir / row.mask_path)
        return Sample(sample_id=row.sample_id, image=None, target=mask, metadata=dict(row.metadata))
