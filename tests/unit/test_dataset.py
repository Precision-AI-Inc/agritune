# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.data.dataset."""

from pathlib import Path

from PIL import Image

from precisionai.agritune.data.dataset import ManifestDataset, SegmentationDataset
from tests.fixtures.manifest_factory import build_manifest


def test_manifest_dataset_length(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path)
    dataset = ManifestDataset(manifest_path)
    assert len(dataset) == 4


def test_manifest_dataset_getitem_returns_sample(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path)
    dataset = ManifestDataset(manifest_path)
    sample = dataset[0]
    assert sample.sample_id == "sample-0"
    assert isinstance(sample.image, Image.Image)
    assert sample.image.mode == "RGB"
    assert sample.metadata == {"field_id": "field-a"}


def test_manifest_dataset_filters_by_sample_ids(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path)
    dataset = ManifestDataset(manifest_path, sample_ids=["sample-0", "sample-2"])
    assert len(dataset) == 2
    assert {dataset[i].sample_id for i in range(len(dataset))} == {"sample-0", "sample-2"}


def test_manifest_dataset_satisfies_segmentation_dataset_protocol(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path)
    dataset = ManifestDataset(manifest_path)
    assert isinstance(dataset, SegmentationDataset)


def test_load_target_never_sets_image(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path)
    dataset = ManifestDataset(manifest_path)
    sample = dataset.load_target(0)
    assert sample.image is None
    assert sample.sample_id == "sample-0"
    assert isinstance(sample.target, Image.Image)
    assert sample.metadata == {"field_id": "field-a"}


def test_load_target_matches_full_getitem_mask(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path)
    dataset = ManifestDataset(manifest_path)
    full = dataset[0]
    target_only = dataset.load_target(0)
    assert list(full.target.getdata()) == list(target_only.target.getdata())
