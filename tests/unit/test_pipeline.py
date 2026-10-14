# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.augmentations.image.pipeline.

Phase 3 definition of done: applying the same augmentation seed must generate exactly the same
transformed image and mask. The pipeline is built on albumentations; these tests exercise the
public ``ImageAugmentationPipeline``/``prepare_sample`` API rather than any internal transform
function, since the engine swap means there is no longer a per-transform function to call
directly.
"""

import numpy as np
import pytest

from precisionai.agritune.augmentations.image.pipeline import (
    AugmentationMode,
    AugmentationPipelineConfig,
    GeometricConfig,
    ImageAugmentationPipeline,
    PhotometricConfig,
    prepare_sample,
    prepare_target_only,
)
from precisionai.agritune.augmentations.image.seeding import derive_offline_seed
from precisionai.agritune.features.keys import hash_augmentation
from precisionai.agritune.schemas.samples import Sample
from tests.fixtures.image_factory import make_image, make_mask


def _sample() -> Sample:
    return Sample(sample_id="s1", image=make_image((8, 6)), target=make_mask((8, 6), fill_value=2))


def test_default_pipeline_is_a_no_op() -> None:
    sample = _sample()
    pipeline = ImageAugmentationPipeline()
    prepared = pipeline.apply(sample, seed=0)
    assert np.array_equal(np.array(prepared.image), np.array(sample.image))
    assert np.array_equal(np.array(prepared.target), np.array(sample.target))
    assert prepared.augmentation_metadata is not None
    assert prepared.augmentation_metadata.transforms == []


def test_geometric_resize_resizes_both_image_and_mask() -> None:
    sample = _sample()
    config = AugmentationPipelineConfig(geometric=GeometricConfig(resize=(4, 3)))
    prepared = ImageAugmentationPipeline(config).apply(sample, seed=0)
    assert prepared.image.size == (4, 3)
    assert prepared.target.size == (4, 3)


def test_resize_mask_uses_nearest_and_preserves_label_values() -> None:
    sample = Sample(sample_id="s1", image=make_image((8, 6)), target=make_mask((8, 6), fill_value=3))
    config = AugmentationPipelineConfig(geometric=GeometricConfig(resize=(4, 3)))
    prepared = ImageAugmentationPipeline(config).apply(sample, seed=0)
    assert set(np.array(prepared.target).ravel().tolist()) == {3}


def test_same_seed_produces_identical_output() -> None:
    sample = _sample()
    config = AugmentationPipelineConfig(
        geometric=GeometricConfig(horizontal_flip_probability=0.5, rotation_max_degrees=10.0),
        photometric=PhotometricConfig(brightness_range=(0.7, 1.3), noise_probability=0.5),
    )
    pipeline = ImageAugmentationPipeline(config)

    first = pipeline.apply(sample, seed=123)
    second = pipeline.apply(sample, seed=123)

    assert np.array_equal(np.array(first.image), np.array(second.image))
    assert np.array_equal(np.array(first.target), np.array(second.target))
    assert first.augmentation_metadata is not None
    assert second.augmentation_metadata is not None
    assert first.augmentation_metadata.transforms == second.augmentation_metadata.transforms


def test_every_configured_transform_is_applied() -> None:
    sample = _sample()
    config = AugmentationPipelineConfig(
        geometric=GeometricConfig(
            random_crop=(4, 3),
            vertical_flip_probability=1.0,
            rotation_max_degrees=5.0,
        ),
        photometric=PhotometricConfig(
            saturation_range=(0.8, 1.2),
            hue_max_degrees=10.0,
            blur_probability=1.0,
            noise_probability=1.0,
        ),
    )
    prepared = ImageAugmentationPipeline(config).apply(sample, seed=0)
    assert prepared.augmentation_metadata is not None
    applied_names = {record.name for record in prepared.augmentation_metadata.transforms}
    # saturation + hue are fused into one ColorJitter step by the albumentations engine.
    assert applied_names == {
        "RandomCrop",
        "VerticalFlip",
        "Rotate",
        "ColorJitter",
        "GaussianBlur",
        "GaussNoise",
    }
    assert prepared.image.size == (4, 3)
    assert prepared.target.size == (4, 3)


def test_random_crop_larger_than_the_image_pads_instead_of_raising() -> None:
    sample = _sample()  # 8x6
    config = AugmentationPipelineConfig(geometric=GeometricConfig(random_crop=(10, 8)))
    prepared = ImageAugmentationPipeline(config).apply(sample, seed=0)
    assert prepared.image.size == (10, 8)
    assert prepared.target.size == (10, 8)


def test_different_seed_can_produce_different_output() -> None:
    sample = _sample()
    config = AugmentationPipelineConfig(geometric=GeometricConfig(rotation_max_degrees=30.0))
    pipeline = ImageAugmentationPipeline(config)

    first = pipeline.apply(sample, seed=1)
    second = pipeline.apply(sample, seed=2)

    assert first.augmentation_metadata is not None
    assert second.augmentation_metadata is not None
    assert first.augmentation_metadata.transforms != second.augmentation_metadata.transforms


def test_mask_never_receives_photometric_transforms() -> None:
    sample = _sample()
    config = AugmentationPipelineConfig(
        photometric=PhotometricConfig(brightness_range=(0.2, 0.2), contrast_range=(2.0, 2.0))
    )
    prepared = ImageAugmentationPipeline(config).apply(sample, seed=0)
    assert np.array_equal(np.array(prepared.target), np.array(sample.target))
    assert not np.array_equal(np.array(prepared.image), np.array(sample.image))


def test_channel_dropout_zeros_one_channel_and_never_touches_the_mask() -> None:
    sample = _sample()
    config = AugmentationPipelineConfig(photometric=PhotometricConfig(channel_dropout_probability=1.0))
    prepared = ImageAugmentationPipeline(config).apply(sample, seed=0)

    array = np.array(prepared.image)
    zeroed_channels = [channel for channel in range(3) if np.all(array[:, :, channel] == 0)]
    assert len(zeroed_channels) == 1
    assert np.array_equal(np.array(prepared.target), np.array(sample.target))


def test_gsd_jitter_degrades_resolution_while_preserving_size() -> None:
    sample = _sample()
    config = AugmentationPipelineConfig(photometric=PhotometricConfig(gsd_jitter_probability=1.0))
    prepared = ImageAugmentationPipeline(config).apply(sample, seed=0)

    assert prepared.image.size == sample.image.size
    assert not np.array_equal(np.array(prepared.image), np.array(sample.image))


def test_vegetation_index_jitter_shifts_red_and_green_but_not_blue() -> None:
    sample = _sample()
    config = AugmentationPipelineConfig(
        photometric=PhotometricConfig(
            vegetation_index_jitter_probability=1.0, vegetation_index_jitter_range=(20.0, 20.0)
        )
    )
    prepared = ImageAugmentationPipeline(config).apply(sample, seed=0)

    original, augmented = np.array(sample.image), np.array(prepared.image)
    assert not np.array_equal(original[:, :, 0], augmented[:, :, 0])
    assert not np.array_equal(original[:, :, 1], augmented[:, :, 1])
    assert np.array_equal(original[:, :, 2], augmented[:, :, 2])


def test_seasonal_color_shift_changes_hue_but_not_geometry() -> None:
    sample = _sample()
    config = AugmentationPipelineConfig(
        photometric=PhotometricConfig(seasonal_color_shift_probability=1.0, seasonal_hue_shift_degrees=30.0)
    )
    prepared = ImageAugmentationPipeline(config).apply(sample, seed=0)

    assert prepared.image.size == sample.image.size
    assert not np.array_equal(np.array(prepared.image), np.array(sample.image))
    assert np.array_equal(np.array(prepared.target), np.array(sample.target))


def test_prepare_sample_none_mode_passes_through_unchanged() -> None:
    sample = _sample()
    pipeline = ImageAugmentationPipeline(AugmentationPipelineConfig(geometric=GeometricConfig(rotation_max_degrees=30)))
    prepared = prepare_sample(sample, mode=AugmentationMode.NONE, pipeline=pipeline, global_seed=0)
    assert prepared.image is sample.image
    assert prepared.target is sample.target
    assert prepared.augmentation_metadata is None


def test_prepare_sample_offline_mode_matches_pipeline_with_derived_seed() -> None:
    sample = _sample()
    config = AugmentationPipelineConfig(geometric=GeometricConfig(rotation_max_degrees=30))
    pipeline = ImageAugmentationPipeline(config)

    prepared = prepare_sample(sample, mode=AugmentationMode.OFFLINE, pipeline=pipeline, global_seed=5, variant=2)
    expected_seed = derive_offline_seed(global_seed=5, sample_id=sample.sample_id, variant=2)
    expected = pipeline.apply(sample, seed=expected_seed)

    assert np.array_equal(np.array(prepared.image), np.array(expected.image))
    assert prepared.augmentation_metadata is not None
    assert prepared.augmentation_metadata.seed == expected_seed


def test_prepare_sample_hybrid_mode_matches_offline_when_probability_is_zero() -> None:
    sample = _sample()
    config = AugmentationPipelineConfig(geometric=GeometricConfig(rotation_max_degrees=30))
    pipeline = ImageAugmentationPipeline(config)

    hybrid = prepare_sample(
        sample,
        mode=AugmentationMode.HYBRID,
        pipeline=pipeline,
        global_seed=5,
        variant=2,
        epoch=3,
        hybrid_online_probability=0.0,
    )
    offline = prepare_sample(sample, mode=AugmentationMode.OFFLINE, pipeline=pipeline, global_seed=5, variant=2)

    assert hybrid.augmentation_metadata is not None
    assert offline.augmentation_metadata is not None
    assert hybrid.augmentation_metadata.seed == offline.augmentation_metadata.seed


def test_prepare_sample_hybrid_mode_matches_online_when_probability_is_one() -> None:
    sample = _sample()
    config = AugmentationPipelineConfig(geometric=GeometricConfig(rotation_max_degrees=30))
    pipeline = ImageAugmentationPipeline(config)

    hybrid = prepare_sample(
        sample,
        mode=AugmentationMode.HYBRID,
        pipeline=pipeline,
        global_seed=5,
        epoch=3,
        hybrid_online_probability=1.0,
    )
    online = prepare_sample(sample, mode=AugmentationMode.ONLINE, pipeline=pipeline, global_seed=5, epoch=3)

    assert hybrid.augmentation_metadata is not None
    assert online.augmentation_metadata is not None
    assert hybrid.augmentation_metadata.seed == online.augmentation_metadata.seed


def test_prepare_sample_online_mode_varies_by_epoch() -> None:
    sample = _sample()
    config = AugmentationPipelineConfig(geometric=GeometricConfig(rotation_max_degrees=30))
    pipeline = ImageAugmentationPipeline(config)

    epoch_0 = prepare_sample(sample, mode=AugmentationMode.ONLINE, pipeline=pipeline, global_seed=0, epoch=0)
    epoch_1 = prepare_sample(sample, mode=AugmentationMode.ONLINE, pipeline=pipeline, global_seed=0, epoch=1)

    assert epoch_0.augmentation_metadata is not None
    assert epoch_1.augmentation_metadata is not None
    assert epoch_0.augmentation_metadata.seed != epoch_1.augmentation_metadata.seed


# --- apply_to_mask / prepare_target_only: feature_provider=cached's image-free batch-building path ---
#
# The critical invariant under test throughout this section: a cache key is
# hash_augmentation(sample.augmentation_metadata), and hash_augmentation hashes the *entire*
# transform list — including photometric transforms, which never touch a mask at all. So
# apply_to_mask must reproduce byte-identical resolved parameters for every transform, geometric
# and photometric alike, purely from the mask and the seed, or a mask-only-built batch's cache
# lookup would silently miss the exact same entry apply() cached it under.


def _full_config() -> AugmentationPipelineConfig:
    return AugmentationPipelineConfig(
        geometric=GeometricConfig(
            resize=(8, 6),
            random_crop=(6, 4),
            horizontal_flip_probability=1.0,
            vertical_flip_probability=1.0,
            rotation_max_degrees=25.0,
        ),
        photometric=PhotometricConfig(
            brightness_range=(0.6, 1.4),
            contrast_range=(0.6, 1.4),
            saturation_range=(0.6, 1.4),
            hue_max_degrees=15.0,
            blur_probability=1.0,
            noise_probability=1.0,
            channel_dropout_probability=1.0,
        ),
    )


@pytest.mark.parametrize("seed", [0, 1, 123, 999])
def test_apply_to_mask_reproduces_apply_s_mask_pixels(seed: int) -> None:
    sample = _sample()
    pipeline = ImageAugmentationPipeline(_full_config())

    prepared = pipeline.apply(sample, seed=seed)
    mask_only, _ = pipeline.apply_to_mask(sample.target, seed=seed)

    assert np.array_equal(np.array(prepared.target), np.array(mask_only))


@pytest.mark.parametrize("seed", [0, 1, 123, 999])
def test_apply_to_mask_reproduces_a_hash_matching_cache_key(seed: int) -> None:
    """The decisive test: even with photometric transforms active (which never touch the mask),
    the mask-only record must hash identically to the real image+mask record — otherwise
    feature_provider=cached lookups from a mask-only-built batch would miss the cache entirely."""
    sample = _sample()
    pipeline = ImageAugmentationPipeline(_full_config())

    prepared = pipeline.apply(sample, seed=seed)
    _, mask_only_record = pipeline.apply_to_mask(sample.target, seed=seed)

    assert hash_augmentation(prepared.augmentation_metadata) == hash_augmentation(mask_only_record)


def test_apply_to_mask_matches_apply_with_resize_only() -> None:
    """The trivial (but common) case: only a deterministic resize is configured, no randomness."""
    sample = _sample()
    pipeline = ImageAugmentationPipeline(AugmentationPipelineConfig(geometric=GeometricConfig(resize=(4, 3))))

    prepared = pipeline.apply(sample, seed=0)
    mask_only, record = pipeline.apply_to_mask(sample.target, seed=0)

    assert mask_only.size == (4, 3)
    assert np.array_equal(np.array(prepared.target), np.array(mask_only))
    assert hash_augmentation(prepared.augmentation_metadata) == hash_augmentation(record)


def test_apply_to_mask_matches_apply_with_no_transforms_configured() -> None:
    sample = _sample()
    pipeline = ImageAugmentationPipeline()

    prepared = pipeline.apply(sample, seed=0)
    mask_only, record = pipeline.apply_to_mask(sample.target, seed=0)

    assert np.array_equal(np.array(prepared.target), np.array(mask_only))
    assert hash_augmentation(prepared.augmentation_metadata) == hash_augmentation(record)
    assert record.transforms == []


def test_prepare_target_only_none_mode_passes_mask_through_unchanged() -> None:
    sample = _sample()
    pipeline = ImageAugmentationPipeline(AugmentationPipelineConfig(geometric=GeometricConfig(rotation_max_degrees=30)))

    mask, record = prepare_target_only(
        sample.target, sample_id=sample.sample_id, mode=AugmentationMode.NONE, pipeline=pipeline, global_seed=0
    )

    assert mask is sample.target
    assert record is None


def test_prepare_target_only_offline_mode_matches_prepare_sample() -> None:
    sample = _sample()
    pipeline = ImageAugmentationPipeline(_full_config())

    prepared = prepare_sample(sample, mode=AugmentationMode.OFFLINE, pipeline=pipeline, global_seed=5, variant=2)
    mask, record = prepare_target_only(
        sample.target,
        sample_id=sample.sample_id,
        mode=AugmentationMode.OFFLINE,
        pipeline=pipeline,
        global_seed=5,
        variant=2,
    )

    assert np.array_equal(np.array(prepared.target), np.array(mask))
    assert prepared.augmentation_metadata is not None
    assert record is not None
    assert hash_augmentation(prepared.augmentation_metadata) == hash_augmentation(record)


@pytest.mark.parametrize("mode", [AugmentationMode.ONLINE, AugmentationMode.HYBRID])
def test_prepare_target_only_rejects_online_and_hybrid(mode: AugmentationMode) -> None:
    sample = _sample()
    pipeline = ImageAugmentationPipeline(_full_config())
    with pytest.raises(ValueError, match="supports mode 'none' or 'offline' only"):
        prepare_target_only(sample.target, sample_id=sample.sample_id, mode=mode, pipeline=pipeline, global_seed=0)
