# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the segmentation services.

Used by :mod:`training_service`, :mod:`evaluation_service`, and :mod:`prediction_service`:
building batches, decoders, and a
:class:`~precisionai.agritune.features.provider.CachedFeatureProvider` keyed consistently with
what ``agritune features build`` wrote.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from precisionai.agritune.data.dataset import ManifestDataset
from precisionai.agritune.data.manifest import ManifestRow, load_manifest
from precisionai.agritune.features.keys import EncoderFingerprint, hash_image_bytes
from precisionai.agritune.features.provider import CachedFeatureProvider
from precisionai.agritune.schemas.protocols import FeatureStore
from precisionai.agritune.schemas.samples import PreparedSample
from precisionai.agritune.tasks.segmentation.decoders.linear import LinearProbeDecoder
from precisionai.agritune.tasks.segmentation.decoders.token_fpn import TokenFPNDecoder
from precisionai.agritune.training.evaluator import TrainingBatch


def mask_to_target_tensor(mask: Any) -> torch.Tensor:
    """Convert a decoded mask image to an integer class-index tensor, shape ``(H, W)``."""
    return torch.from_numpy(np.array(mask)).long()


def build_training_batches(dataset: ManifestDataset, *, batch_size: int) -> list[TrainingBatch]:
    """Chunk a dataset into :class:`TrainingBatch` objects of at most ``batch_size`` samples."""
    batches = []
    for start in range(0, len(dataset), batch_size):
        chunk = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        samples = [PreparedSample(sample_id=sample.sample_id, image=sample.image, target=None) for sample in chunk]
        targets = torch.stack([mask_to_target_tensor(sample.target) for sample in chunk])
        batches.append(TrainingBatch(samples=samples, targets=targets))
    return batches


def build_prediction_batches(dataset: ManifestDataset, *, batch_size: int) -> list[list[PreparedSample]]:
    """Chunk a dataset into sample batches with no target tensor required (for inference)."""
    batches = []
    for start in range(0, len(dataset), batch_size):
        chunk = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        batches.append(
            [PreparedSample(sample_id=sample.sample_id, image=sample.image, target=None) for sample in chunk]
        )
    return batches


def build_decoder(
    name: str, *, patch_dim: int, cls_dim: int | None, num_classes: int, output_size: tuple[int, int]
) -> nn.Module:
    """Construct the named decoder (``"linear"`` or ``"token_fpn"``)."""
    if name == "linear":
        return LinearProbeDecoder(patch_dim=patch_dim, num_classes=num_classes, output_size=output_size)
    if name == "token_fpn":
        return TokenFPNDecoder(patch_dim=patch_dim, num_classes=num_classes, output_size=output_size, cls_dim=cls_dim)
    raise ValueError(f"unsupported decoder: {name!r}")


def probe_feature_dims(provider: CachedFeatureProvider, sample: PreparedSample) -> tuple[int, int | None]:
    """Return ``(patch_dim, cls_dim)`` observed from one sample's cached features."""
    features = provider.get_features([sample])
    patch_dim = features.patch_tokens.shape[-1]
    cls_dim = features.cls_tokens.shape[-1] if features.cls_tokens is not None else None
    return patch_dim, cls_dim


def build_cached_feature_provider(
    manifest_path: str, *, store: FeatureStore, encoder_fingerprint: EncoderFingerprint
) -> tuple[CachedFeatureProvider, list[ManifestRow]]:
    """Build a :class:`CachedFeatureProvider` keyed identically to ``agritune features build``.

    Returns
    -------
    tuple[CachedFeatureProvider, list[ManifestRow]]
        The provider, plus the parsed manifest rows (callers typically need these for splitting).
    """
    rows = load_manifest(manifest_path)
    base_dir = Path(manifest_path).parent
    rows_by_sample_id = {row.sample_id: row for row in rows}

    def image_hash_fn(sample: PreparedSample) -> str:
        # The feature store was built by feature_service.build_features using file-content
        # hashes; the provider must derive the identical key here to find them.
        row = rows_by_sample_id[sample.sample_id]
        return hash_image_bytes((base_dir / row.image_path).read_bytes())

    provider = CachedFeatureProvider(store, encoder_fingerprint=encoder_fingerprint, image_hash_fn=image_hash_fn)
    return provider, rows
