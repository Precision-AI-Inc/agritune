# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.schemas.augmentation."""

from precisionai.agritune.schemas.augmentation import AugmentationRecord, TransformRecord


def test_augmentation_record_defaults_to_no_transforms() -> None:
    record = AugmentationRecord(seed=1)
    assert record.transforms == []


def test_augmentation_record_carries_transform_list() -> None:
    transforms = [TransformRecord(name="horizontal_flip", params={"applied": True})]
    record = AugmentationRecord(seed=1, transforms=transforms)
    assert record.transforms[0].name == "horizontal_flip"
    assert record.transforms[0].params == {"applied": True}


def test_transform_record_defaults_to_empty_params() -> None:
    transform = TransformRecord(name="resize")
    assert transform.params == {}
