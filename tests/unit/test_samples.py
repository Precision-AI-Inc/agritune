# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.schemas.samples."""

from precisionai.agritune.schemas.augmentation import AugmentationRecord
from precisionai.agritune.schemas.samples import PreparedSample, Sample


def test_sample_metadata_defaults_to_empty_dict() -> None:
    sample = Sample(sample_id="s1", image=None, target=None)
    assert sample.metadata == {}


def test_sample_accepts_agricultural_metadata() -> None:
    sample = Sample(
        sample_id="s1",
        image=None,
        target=None,
        metadata={"farm_id": "f1", "field_id": "fd1", "crop": "corn"},
    )
    assert sample.metadata["crop"] == "corn"


def test_sample_metadata_is_not_mandatory() -> None:
    sample = Sample(sample_id="s1", image=None, target=None)
    assert "farm_id" not in sample.metadata


def test_prepared_sample_augmentation_metadata_defaults_to_none() -> None:
    prepared = PreparedSample(sample_id="s1", image=None, target=None)
    assert prepared.augmentation_metadata is None


def test_prepared_sample_carries_sample_id() -> None:
    record = AugmentationRecord(seed=42)
    prepared = PreparedSample(sample_id="s1", image="img", target="mask", augmentation_metadata=record)
    assert prepared.sample_id == "s1"
    assert prepared.augmentation_metadata is not None
    assert prepared.augmentation_metadata.seed == 42
