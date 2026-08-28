# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.augmentations.image.transforms."""

import random

import numpy as np

from precisionai.agritune.augmentations.image import transforms
from tests.fixtures.image_factory import make_image, make_mask


def test_resize_produces_exact_target_size() -> None:
    image, mask = make_image((8, 6)), make_mask((8, 6))
    resized_image, resized_mask, record = transforms.resize(image, mask, (4, 3))
    assert resized_image.size == (4, 3)
    assert resized_mask.size == (4, 3)
    assert record.params["size"] == [4, 3]


def test_resize_mask_uses_nearest_and_preserves_label_values() -> None:
    image, mask = make_image((8, 6)), make_mask((8, 6), fill_value=3)
    _, resized_mask, _ = transforms.resize(image, mask, (4, 3))
    assert set(np.array(resized_mask).ravel().tolist()) == {3}


def test_random_crop_produces_exact_crop_size() -> None:
    image, mask = make_image((8, 6)), make_mask((8, 6))
    cropped_image, cropped_mask, record = transforms.random_crop(image, mask, (4, 3), random.Random(0))
    assert cropped_image.size == (4, 3)
    assert cropped_mask.size == (4, 3)
    assert "box" in record.params


def test_random_crop_is_deterministic_given_same_rng_state() -> None:
    image, mask = make_image((8, 6)), make_mask((8, 6))
    first_image, _, first_record = transforms.random_crop(image, mask, (4, 3), random.Random(42))
    second_image, _, second_record = transforms.random_crop(image, mask, (4, 3), random.Random(42))
    assert first_record.params == second_record.params
    assert np.array_equal(np.array(first_image), np.array(second_image))


def test_horizontal_flip_probability_one_always_flips() -> None:
    image, mask = make_image((8, 6)), make_mask((8, 6))
    flipped_image, _, record = transforms.horizontal_flip(image, mask, random.Random(0), probability=1.0)
    assert record.params["applied"] is True
    assert np.array_equal(np.array(flipped_image), np.array(image)[:, ::-1, :])


def test_horizontal_flip_probability_zero_never_flips() -> None:
    image, mask = make_image((8, 6)), make_mask((8, 6))
    flipped_image, flipped_mask, record = transforms.horizontal_flip(image, mask, random.Random(0), probability=0.0)
    assert record.params["applied"] is False
    assert np.array_equal(np.array(flipped_image), np.array(image))
    assert np.array_equal(np.array(flipped_mask), np.array(mask))


def test_vertical_flip_probability_one_always_flips() -> None:
    image, mask = make_image((8, 6)), make_mask((8, 6))
    flipped_image, _, record = transforms.vertical_flip(image, mask, random.Random(0), probability=1.0)
    assert record.params["applied"] is True
    assert np.array_equal(np.array(flipped_image), np.array(image)[::-1, :, :])


def test_rotation_zero_degrees_is_a_no_op_angle() -> None:
    image, mask = make_image((8, 6)), make_mask((8, 6))
    _, _, record = transforms.rotation(image, mask, max_degrees=0.0, rng=random.Random(0))
    assert record.params["degrees"] == 0.0


def test_brightness_returns_factor_within_range() -> None:
    image = make_image((8, 6))
    _, record = transforms.brightness(image, (0.5, 1.5), random.Random(0))
    assert 0.5 <= record.params["factor"] <= 1.5


def test_contrast_returns_factor_within_range() -> None:
    image = make_image((8, 6))
    _, record = transforms.contrast(image, (0.5, 1.5), random.Random(0))
    assert 0.5 <= record.params["factor"] <= 1.5


def test_saturation_returns_factor_within_range() -> None:
    image = make_image((8, 6))
    _, record = transforms.saturation(image, (0.5, 1.5), random.Random(0))
    assert 0.5 <= record.params["factor"] <= 1.5


def test_hue_zero_shift_records_zero_degrees() -> None:
    image = make_image((8, 6))
    _, record = transforms.hue(image, max_shift_degrees=0.0, rng=random.Random(0))
    assert record.params["degrees"] == 0.0


def test_blur_returns_radius_within_range() -> None:
    image = make_image((8, 6))
    blurred, record = transforms.blur(image, (1.0, 2.0), random.Random(0))
    assert 1.0 <= record.params["radius"] <= 2.0
    assert blurred.size == image.size


def test_noise_returns_std_within_range_and_preserves_size() -> None:
    image = make_image((8, 6))
    noisy, record = transforms.noise(image, (0.01, 0.05), random.Random(0))
    assert 0.01 <= record.params["std"] <= 0.05
    assert noisy.size == image.size
