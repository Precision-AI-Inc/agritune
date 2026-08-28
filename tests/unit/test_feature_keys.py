# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.features.keys."""

from precisionai.agritune.features.keys import (
    EncoderFingerprint,
    compute_feature_key,
    hash_augmentation,
    hash_image_bytes,
)
from precisionai.agritune.schemas.augmentation import AugmentationRecord, TransformRecord

_FINGERPRINT = EncoderFingerprint(model="pai-embedding", revision=None, preprocessing="resize=224")


def test_hash_image_bytes_is_deterministic() -> None:
    assert hash_image_bytes(b"abc") == hash_image_bytes(b"abc")


def test_hash_image_bytes_differs_for_different_content() -> None:
    assert hash_image_bytes(b"abc") != hash_image_bytes(b"abd")


def test_hash_augmentation_none_is_stable_sentinel() -> None:
    assert hash_augmentation(None) == "none"


def test_hash_augmentation_is_deterministic() -> None:
    record = AugmentationRecord(seed=1, transforms=[TransformRecord(name="flip", params={"applied": True})])
    assert hash_augmentation(record) == hash_augmentation(record)


def test_hash_augmentation_differs_by_seed() -> None:
    first = AugmentationRecord(seed=1)
    second = AugmentationRecord(seed=2)
    assert hash_augmentation(first) != hash_augmentation(second)


def test_hash_augmentation_differs_by_transform_params() -> None:
    first = AugmentationRecord(seed=1, transforms=[TransformRecord(name="rotation", params={"degrees": 5.0})])
    second = AugmentationRecord(seed=1, transforms=[TransformRecord(name="rotation", params={"degrees": 10.0})])
    assert hash_augmentation(first) != hash_augmentation(second)


def test_compute_feature_key_is_deterministic() -> None:
    key_a = compute_feature_key(
        sample_id="s1", image_hash="h1", augmentation_fingerprint="none", encoder_fingerprint=_FINGERPRINT
    )
    key_b = compute_feature_key(
        sample_id="s1", image_hash="h1", augmentation_fingerprint="none", encoder_fingerprint=_FINGERPRINT
    )
    assert key_a == key_b


def test_compute_feature_key_changes_with_sample_id() -> None:
    key_a = compute_feature_key(
        sample_id="s1", image_hash="h1", augmentation_fingerprint="none", encoder_fingerprint=_FINGERPRINT
    )
    key_b = compute_feature_key(
        sample_id="s2", image_hash="h1", augmentation_fingerprint="none", encoder_fingerprint=_FINGERPRINT
    )
    assert key_a != key_b


def test_compute_feature_key_changes_with_image_hash() -> None:
    key_a = compute_feature_key(
        sample_id="s1", image_hash="h1", augmentation_fingerprint="none", encoder_fingerprint=_FINGERPRINT
    )
    key_b = compute_feature_key(
        sample_id="s1", image_hash="h2", augmentation_fingerprint="none", encoder_fingerprint=_FINGERPRINT
    )
    assert key_a != key_b


def test_compute_feature_key_changes_with_augmentation_fingerprint() -> None:
    key_a = compute_feature_key(
        sample_id="s1", image_hash="h1", augmentation_fingerprint="none", encoder_fingerprint=_FINGERPRINT
    )
    key_b = compute_feature_key(
        sample_id="s1", image_hash="h1", augmentation_fingerprint="other", encoder_fingerprint=_FINGERPRINT
    )
    assert key_a != key_b


def test_compute_feature_key_changes_with_encoder_model() -> None:
    other_fingerprint = EncoderFingerprint(model="other-model", revision=None, preprocessing="resize=224")
    key_a = compute_feature_key(
        sample_id="s1", image_hash="h1", augmentation_fingerprint="none", encoder_fingerprint=_FINGERPRINT
    )
    key_b = compute_feature_key(
        sample_id="s1", image_hash="h1", augmentation_fingerprint="none", encoder_fingerprint=other_fingerprint
    )
    assert key_a != key_b


def test_compute_feature_key_changes_with_feature_schema_version() -> None:
    key_a = compute_feature_key(
        sample_id="s1",
        image_hash="h1",
        augmentation_fingerprint="none",
        encoder_fingerprint=_FINGERPRINT,
        feature_schema_version=1,
    )
    key_b = compute_feature_key(
        sample_id="s1",
        image_hash="h1",
        augmentation_fingerprint="none",
        encoder_fingerprint=_FINGERPRINT,
        feature_schema_version=2,
    )
    assert key_a != key_b
