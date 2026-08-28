# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.features.integrity."""

from pathlib import Path

import torch

from precisionai.agritune.features.integrity import verify_store
from precisionai.agritune.features.store import DirectoryFeatureStore
from precisionai.agritune.schemas.features import EncoderFeatures


def _features() -> EncoderFeatures:
    return EncoderFeatures(
        patch_tokens=torch.randn(1, 4, 8),
        cls_tokens=None,
        patch_grid=torch.tensor([[2, 2]]),
        valid_patch_mask=None,
        image_sizes=[(224, 224)],
        encoder_model="fake",
        encoder_revision=None,
    )


def test_verify_store_reports_valid_for_clean_store(tmp_path: Path) -> None:
    store = DirectoryFeatureStore(tmp_path)
    store.write("key1", _features())
    store.write("key2", _features())

    report = verify_store(store)

    assert report.total == 2
    assert report.is_valid
    assert report.corrupted_keys == []


def test_verify_store_reports_corrupted_keys(tmp_path: Path) -> None:
    store = DirectoryFeatureStore(tmp_path)
    store.write("key1", _features())
    store.write("key2", _features())
    with (tmp_path / "key2.safetensors").open("r+b") as handle:
        handle.seek(0)
        handle.write(b"\x00" * 16)

    report = verify_store(store)

    assert not report.is_valid
    assert report.corrupted_keys == ["key2"]


def test_verify_store_empty_store(tmp_path: Path) -> None:
    store = DirectoryFeatureStore(tmp_path)
    report = verify_store(store)
    assert report.total == 0
    assert report.is_valid
