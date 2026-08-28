# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.features.manifest."""

from pathlib import Path

import torch

from precisionai.agritune.features.manifest import FeatureManifest
from precisionai.agritune.features.store import DirectoryFeatureStore, FeatureSummary
from precisionai.agritune.schemas.features import EncoderFeatures


def _features(*, patch_dim: int, encoder_model: str) -> EncoderFeatures:
    return EncoderFeatures(
        patch_tokens=torch.randn(1, 4, patch_dim),
        cls_tokens=None,
        patch_grid=torch.tensor([[2, 2]]),
        valid_patch_mask=None,
        image_sizes=[(224, 224)],
        encoder_model=encoder_model,
        encoder_revision=None,
    )


def test_empty_store_produces_empty_manifest(tmp_path: Path) -> None:
    store = DirectoryFeatureStore(tmp_path)
    manifest = FeatureManifest.from_store(store)
    assert len(manifest) == 0


def test_manifest_reflects_every_stored_entry(tmp_path: Path) -> None:
    store = DirectoryFeatureStore(tmp_path)
    store.write("key1", _features(patch_dim=8, encoder_model="pai-embedding"))
    store.write("key2", _features(patch_dim=8, encoder_model="pai-embedding"))
    manifest = FeatureManifest.from_store(store)
    assert len(manifest) == 2


def test_encoder_models_counts_distinct_models(tmp_path: Path) -> None:
    store = DirectoryFeatureStore(tmp_path)
    store.write("key1", _features(patch_dim=8, encoder_model="model-a"))
    store.write("key2", _features(patch_dim=8, encoder_model="model-b"))
    manifest = FeatureManifest.from_store(store)
    assert manifest.encoder_models() == {"model-a": 1, "model-b": 1}


def test_patch_dims_counts_distinct_dimensions() -> None:
    manifest = FeatureManifest(
        entries=[
            _summary(patch_dim=8),
            _summary(patch_dim=8),
            _summary(patch_dim=16),
        ]
    )
    assert manifest.patch_dims() == {8: 2, 16: 1}


def _summary(*, patch_dim: int) -> FeatureSummary:
    return FeatureSummary(
        key="k",
        encoder_model="m",
        encoder_revision=None,
        patch_dim=patch_dim,
        cls_dim=None,
        num_patches=4,
        image_size=(224, 224),
        checksum="c",
    )
