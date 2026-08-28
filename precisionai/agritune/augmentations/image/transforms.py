# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Geometric and photometric image/mask transforms.

Masks receive only geometric transforms, and always with nearest-neighbor interpolation — resizing
or rotating a label mask with bilinear/bicubic interpolation invents fractional class values that
do not exist in the label space. Images use bilinear (resize/crop, which never need to invent
new geometry beyond translation) or bicubic (rotation) interpolation. Photometric transforms only
ever touch the image.
"""

import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from precisionai.agritune.schemas.augmentation import TransformRecord

_IMAGE_RESIZE_RESAMPLE = Image.Resampling.BILINEAR
_MASK_RESAMPLE = Image.Resampling.NEAREST
_IMAGE_ROTATE_RESAMPLE = Image.Resampling.BICUBIC


def resize(
    image: Image.Image, mask: Image.Image, size: tuple[int, int]
) -> tuple[Image.Image, Image.Image, TransformRecord]:
    """Resize ``image`` (bilinear) and ``mask`` (nearest) to ``(width, height) = size``."""
    resized_image = image.resize(size, resample=_IMAGE_RESIZE_RESAMPLE)
    resized_mask = mask.resize(size, resample=_MASK_RESAMPLE)
    return resized_image, resized_mask, TransformRecord(name="resize", params={"size": list(size)})


def random_crop(
    image: Image.Image, mask: Image.Image, crop_size: tuple[int, int], rng: random.Random
) -> tuple[Image.Image, Image.Image, TransformRecord]:
    """Crop an identical, randomly placed ``(width, height) = crop_size`` box from image and mask."""
    crop_width, crop_height = crop_size
    max_x = max(image.width - crop_width, 0)
    max_y = max(image.height - crop_height, 0)
    left = rng.randint(0, max_x)
    top = rng.randint(0, max_y)
    box = (left, top, left + crop_width, top + crop_height)
    return (
        image.crop(box),
        mask.crop(box),
        TransformRecord(name="random_crop", params={"box": list(box)}),
    )


def horizontal_flip(
    image: Image.Image, mask: Image.Image, rng: random.Random, *, probability: float = 0.5
) -> tuple[Image.Image, Image.Image, TransformRecord]:
    """Flip image and mask left-right with the given probability."""
    applied = rng.random() < probability
    if applied:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return image, mask, TransformRecord(name="horizontal_flip", params={"applied": applied})


def vertical_flip(
    image: Image.Image, mask: Image.Image, rng: random.Random, *, probability: float = 0.5
) -> tuple[Image.Image, Image.Image, TransformRecord]:
    """Flip image and mask top-bottom with the given probability."""
    applied = rng.random() < probability
    if applied:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return image, mask, TransformRecord(name="vertical_flip", params={"applied": applied})


def rotation(
    image: Image.Image, mask: Image.Image, max_degrees: float, rng: random.Random
) -> tuple[Image.Image, Image.Image, TransformRecord]:
    """Rotate image (bicubic) and mask (nearest) by the same angle in ``[-max_degrees, max_degrees]``."""
    angle = rng.uniform(-max_degrees, max_degrees)
    rotated_image = image.rotate(angle, resample=_IMAGE_ROTATE_RESAMPLE, expand=False)
    rotated_mask = mask.rotate(angle, resample=_MASK_RESAMPLE, expand=False)
    return rotated_image, rotated_mask, TransformRecord(name="rotation", params={"degrees": angle})


def brightness(
    image: Image.Image, factor_range: tuple[float, float], rng: random.Random
) -> tuple[Image.Image, TransformRecord]:
    """Randomly adjust brightness; ``factor_range`` is ``(min, max)`` around ``1.0`` = unchanged."""
    factor = rng.uniform(*factor_range)
    adjusted = ImageEnhance.Brightness(image).enhance(factor)
    return adjusted, TransformRecord(name="brightness", params={"factor": factor})


def contrast(
    image: Image.Image, factor_range: tuple[float, float], rng: random.Random
) -> tuple[Image.Image, TransformRecord]:
    """Randomly adjust contrast; ``factor_range`` is ``(min, max)`` around ``1.0`` = unchanged."""
    factor = rng.uniform(*factor_range)
    adjusted = ImageEnhance.Contrast(image).enhance(factor)
    return adjusted, TransformRecord(name="contrast", params={"factor": factor})


def saturation(
    image: Image.Image, factor_range: tuple[float, float], rng: random.Random
) -> tuple[Image.Image, TransformRecord]:
    """Randomly adjust color saturation; ``factor_range`` is ``(min, max)`` around ``1.0`` = unchanged."""
    factor = rng.uniform(*factor_range)
    adjusted = ImageEnhance.Color(image).enhance(factor)
    return adjusted, TransformRecord(name="saturation", params={"factor": factor})


def hue(image: Image.Image, max_shift_degrees: float, rng: random.Random) -> tuple[Image.Image, TransformRecord]:
    """Randomly shift hue by up to ``max_shift_degrees`` (of 360) via the image's HSV encoding."""
    shift_degrees = rng.uniform(-max_shift_degrees, max_shift_degrees)
    shift = round((shift_degrees / 360.0) * 256)
    hsv = np.array(image.convert("HSV"))
    hsv[..., 0] = (hsv[..., 0].astype(np.int32) + shift) % 256
    adjusted = Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert(image.mode)
    return adjusted, TransformRecord(name="hue", params={"degrees": shift_degrees})


def blur(
    image: Image.Image, radius_range: tuple[float, float], rng: random.Random
) -> tuple[Image.Image, TransformRecord]:
    """Apply Gaussian blur with a radius sampled from ``radius_range``."""
    radius = rng.uniform(*radius_range)
    adjusted = image.filter(ImageFilter.GaussianBlur(radius=radius))
    return adjusted, TransformRecord(name="blur", params={"radius": radius})


def noise(
    image: Image.Image, std_range: tuple[float, float], rng: random.Random
) -> tuple[Image.Image, TransformRecord]:
    """Add zero-mean Gaussian pixel noise; ``std_range`` is sampled in normalized ``[0, 1]`` units."""
    std = rng.uniform(*std_range)
    array = np.array(image).astype(np.float32) / 255.0
    generator = np.random.default_rng(rng.randint(0, 2**32 - 1))
    noisy = array + generator.normal(loc=0.0, scale=std, size=array.shape)
    noisy = np.clip(noisy, 0.0, 1.0) * 255.0
    adjusted = Image.fromarray(noisy.astype(np.uint8), mode=image.mode)
    return adjusted, TransformRecord(name="noise", params={"std": std})
