# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""The deterministic image augmentation pipeline: config, orchestration, and mode dispatch.

Built on `albumentations <https://albumentations.ai/>`_'s ``Compose`` as the composition engine.
Geometric transforms (resize/crop/flip/rotate) receive both ``image`` and ``mask`` targets —
images with bilinear/bicubic interpolation, masks always with nearest-neighbor, since resizing or
rotating a label mask with anything else invents fractional class values that do not exist in the
label space. Photometric and domain-specific spectral transforms only ever touch ``image`` —
albumentations' color/noise/blur/channel transforms never declare a ``mask`` target, so this is
enforced by the library itself rather than by manual gating here.

Determinism is driven by ``Compose.set_random_seed(seed)`` before every ``apply()`` call, on a
``Compose`` built once from the configured transform list — the same seed on the same sample and
config always produces byte-identical output. ``TransformRecord.name``/``params`` come straight
from ``Compose(..., save_applied_params=True)``'s ``applied_transforms`` — only transforms that
actually fired (e.g. a flip whose probability didn't trigger is omitted, not recorded as
"not applied") are present, one entry per underlying albumentations transform class.

Supported modes: ``none`` (pass the sample through unchanged), ``offline`` (a fixed number of
precomputed variants per sample, seeded by sample + variant), ``online`` (re-augmented every
epoch, seeded by sample + epoch + occurrence), and ``hybrid`` (mostly reuses an offline variant,
occasionally derives a fresh online seed — see ``agritune_implementation_plan.md`` §6 and
:func:`~precisionai.agritune.augmentations.image.seeding.derive_hybrid_seed`).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import albumentations as alb
import cv2
import numpy as np
from albumentations.core.composition import TransformsSeqType
from PIL import Image

from precisionai.agritune.augmentations.image.seeding import derive_hybrid_seed, derive_offline_seed, derive_online_seed
from precisionai.agritune.schemas.augmentation import AugmentationRecord, TransformRecord
from precisionai.agritune.schemas.samples import PreparedSample, Sample


class AugmentationMode(str, Enum):
    """Supported augmentation modes."""

    NONE = "none"
    OFFLINE = "offline"
    ONLINE = "online"
    HYBRID = "hybrid"


@dataclass
class GeometricConfig:
    """Geometric transform configuration. A ``None``/``0`` value disables that transform.

    Attributes
    ----------
    resize : tuple[int, int] | None
        Target ``(width, height)``, applied first if set.
    random_crop : tuple[int, int] | None
        ``(width, height)`` crop size, applied after resize if set. May exceed the image's own
        size — the crop is padded (constant fill) rather than raising.
    horizontal_flip_probability : float
        Probability in ``[0, 1]``.
    vertical_flip_probability : float
        Probability in ``[0, 1]``.
    rotation_max_degrees : float
        Maximum absolute rotation in degrees; ``0`` disables rotation.
    """

    resize: tuple[int, int] | None = None
    random_crop: tuple[int, int] | None = None
    horizontal_flip_probability: float = 0.0
    vertical_flip_probability: float = 0.0
    rotation_max_degrees: float = 0.0


@dataclass
class PhotometricConfig:
    """Photometric and spectral transform configuration (image only). A ``None``/``0`` disables it.

    ``channel_dropout``/``gsd_jitter``/``vegetation_index_jitter``/``seasonal_color_shift`` are
    agricultural-domain-specific additions: simulating a failed/missing sensor band, variable
    ground-sample-distance across flights, red/green channel-balance variability a true NDVI
    computation would be sensitive to (this RGB-only pipeline has no NIR band, so it is a proxy,
    not literal NDVI), and a seasonal/growth-stage hue shift, respectively.

    Attributes
    ----------
    brightness_range : tuple[float, float] | None
        ``(min, max)`` multiplicative factor around ``1.0``.
    contrast_range : tuple[float, float] | None
        ``(min, max)`` multiplicative factor around ``1.0``.
    saturation_range : tuple[float, float] | None
        ``(min, max)`` multiplicative factor around ``1.0``.
    hue_max_degrees : float
        Maximum absolute hue shift in degrees (of 360); ``0`` disables it.
    blur_probability : float
        Probability in ``[0, 1]`` of applying Gaussian blur.
    blur_radius_range : tuple[float, float]
        Radius (Gaussian sigma) range sampled when blur is applied.
    noise_probability : float
        Probability in ``[0, 1]`` of applying Gaussian pixel noise.
    noise_std_range : tuple[float, float]
        Normalized-``[0, 1]``-scale standard deviation range sampled when noise is applied.
    channel_dropout_probability : float
        Probability in ``[0, 1]`` of zeroing one RGB channel entirely.
    gsd_jitter_probability : float
        Probability in ``[0, 1]`` of a downscale-then-upscale resolution-degrading pass.
    gsd_jitter_scale_range : tuple[float, float]
        Downscale factor range sampled when GSD jitter is applied (smaller = coarser).
    vegetation_index_jitter_probability : float
        Probability in ``[0, 1]`` of an independent red/green channel-value jitter.
    vegetation_index_jitter_range : tuple[float, float]
        ``(min, max)`` additive shift (0-255 scale) sampled independently for red and green.
    seasonal_color_shift_probability : float
        Probability in ``[0, 1]`` of a seasonal/growth-stage hue shift.
    seasonal_hue_shift_degrees : float
        Maximum absolute hue shift, in OpenCV's 0-179 hue-circle units; ``0`` disables it.
    """

    brightness_range: tuple[float, float] | None = None
    contrast_range: tuple[float, float] | None = None
    saturation_range: tuple[float, float] | None = None
    hue_max_degrees: float = 0.0
    blur_probability: float = 0.0
    blur_radius_range: tuple[float, float] = (0.1, 2.0)
    noise_probability: float = 0.0
    noise_std_range: tuple[float, float] = (0.01, 0.05)
    channel_dropout_probability: float = 0.0
    gsd_jitter_probability: float = 0.0
    gsd_jitter_scale_range: tuple[float, float] = (0.5, 0.9)
    vegetation_index_jitter_probability: float = 0.0
    vegetation_index_jitter_range: tuple[float, float] = (-15.0, 15.0)
    seasonal_color_shift_probability: float = 0.0
    seasonal_hue_shift_degrees: float = 0.0


@dataclass
class AugmentationPipelineConfig:
    """Full configuration for one :class:`ImageAugmentationPipeline`."""

    geometric: GeometricConfig = field(default_factory=GeometricConfig)
    photometric: PhotometricConfig = field(default_factory=PhotometricConfig)


def _geometric_transforms(config: GeometricConfig) -> list[alb.BasicTransform]:
    transforms: list[alb.BasicTransform] = []

    if config.resize is not None:
        width, height = config.resize
        transforms.append(
            alb.Resize(height=height, width=width, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST)
        )
    if config.random_crop is not None:
        width, height = config.random_crop
        transforms.append(alb.RandomCrop(height=height, width=width, pad_if_needed=True, p=1.0))
    if config.horizontal_flip_probability > 0:
        transforms.append(alb.HorizontalFlip(p=config.horizontal_flip_probability))
    if config.vertical_flip_probability > 0:
        transforms.append(alb.VerticalFlip(p=config.vertical_flip_probability))
    if config.rotation_max_degrees > 0:
        transforms.append(
            alb.Rotate(
                limit=config.rotation_max_degrees,
                interpolation=cv2.INTER_CUBIC,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                p=1.0,
            )
        )
    return transforms


def _photometric_transforms(config: PhotometricConfig) -> list[alb.BasicTransform]:
    transforms: list[alb.BasicTransform] = []

    color_jitter_enabled = (
        config.brightness_range is not None
        or config.contrast_range is not None
        or config.saturation_range is not None
        or config.hue_max_degrees > 0
    )
    if color_jitter_enabled:
        hue_fraction = config.hue_max_degrees / 360.0
        transforms.append(
            alb.ColorJitter(
                brightness=config.brightness_range or (1.0, 1.0),
                contrast=config.contrast_range or (1.0, 1.0),
                saturation=config.saturation_range or (1.0, 1.0),
                hue=(-hue_fraction, hue_fraction),
                p=1.0,
            )
        )
    if config.blur_probability > 0:
        transforms.append(
            alb.GaussianBlur(blur_limit=0, sigma_limit=config.blur_radius_range, p=config.blur_probability)
        )
    if config.noise_probability > 0:
        transforms.append(
            alb.GaussNoise(std_range=config.noise_std_range, mean_range=(0.0, 0.0), p=config.noise_probability)
        )
    if config.channel_dropout_probability > 0:
        transforms.append(alb.ChannelDropout(channel_drop_range=(1, 1), fill=0.0, p=config.channel_dropout_probability))
    if config.gsd_jitter_probability > 0:
        transforms.append(alb.Downscale(scale_range=config.gsd_jitter_scale_range, p=config.gsd_jitter_probability))
    if config.vegetation_index_jitter_probability > 0:
        jitter_range = config.vegetation_index_jitter_range
        transforms.append(
            alb.RGBShift(
                r_shift_limit=jitter_range,
                g_shift_limit=jitter_range,
                b_shift_limit=(0.0, 0.0),
                p=config.vegetation_index_jitter_probability,
            )
        )
    if config.seasonal_color_shift_probability > 0:
        shift = config.seasonal_hue_shift_degrees
        transforms.append(
            alb.HueSaturationValue(
                hue_shift_limit=(-shift, shift),
                sat_shift_limit=(0.0, 0.0),
                val_shift_limit=(0.0, 0.0),
                p=config.seasonal_color_shift_probability,
            )
        )
    return transforms


def _build_transforms(config: AugmentationPipelineConfig) -> TransformsSeqType:
    # Annotated as TransformsSeqType (not list[alb.BasicTransform]) so this list literal is exactly
    # what Compose.__init__ declares — list's generic parameter is invariant, so a list[BasicTransform]
    # value does not satisfy a list[TransformType] parameter even though BasicTransform is a member
    # of the TransformType union.
    transforms: TransformsSeqType = [
        *_geometric_transforms(config.geometric),
        *_photometric_transforms(config.photometric),
    ]
    return transforms


def _sanitize_params(params: dict[str, Any]) -> dict[str, Any]:
    """Convert numpy arrays in ``params`` (e.g. ``Rotate``'s affine matrix) to plain lists.

    Kept JSON-serializable and directly ``==``-comparable — a raw ``numpy.ndarray`` value makes
    the whole ``dict``'s equality ambiguous, which breaks comparing two ``TransformRecord``s.
    """
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in params.items()}


class ImageAugmentationPipeline:
    """Applies a configured sequence of geometric then photometric transforms, given a seed.

    Parameters
    ----------
    config : AugmentationPipelineConfig | None, optional
        Transform configuration; defaults to every transform disabled (a no-op pipeline).
    """

    def __init__(self, config: AugmentationPipelineConfig | None = None) -> None:
        self._config = config or AugmentationPipelineConfig()
        self._compose = alb.Compose(_build_transforms(self._config), seed=0, save_applied_params=True, strict=True)

    def apply(self, sample: Sample, *, seed: int) -> PreparedSample:
        """Apply this pipeline's configured transforms to one sample, deterministically.

        Parameters
        ----------
        sample : Sample
            The source sample. ``sample.image`` and ``sample.target`` must be ``PIL.Image.Image``.
        seed : int
            Seed controlling every random choice made while applying transforms. The same seed on
            the same sample and config always produces byte-identical output.

        Returns
        -------
        PreparedSample
            The augmented sample, with an :class:`AugmentationRecord` capturing exactly what was
            applied.
        """
        self._compose.set_random_seed(seed)
        result = self._compose(image=np.array(sample.image), mask=np.array(sample.target))

        records = [
            TransformRecord(name=name, params=_sanitize_params(params))
            for name, params in result.get("applied_transforms", [])
        ]

        return PreparedSample(
            sample_id=sample.sample_id,
            image=Image.fromarray(result["image"]),
            target=Image.fromarray(result["mask"]),
            augmentation_metadata=AugmentationRecord(seed=seed, transforms=records),
        )


def prepare_sample(
    sample: Sample,
    *,
    mode: AugmentationMode,
    pipeline: ImageAugmentationPipeline,
    global_seed: int,
    variant: int = 0,
    epoch: int = 0,
    occurrence: int = 0,
    hybrid_online_probability: float = 0.3,
) -> PreparedSample:
    """Dispatch a sample to the right seed derivation for ``mode`` and apply the pipeline.

    Parameters
    ----------
    sample : Sample
        The source sample.
    mode : AugmentationMode
        ``NONE`` passes the sample through with no augmentation and no seed. ``OFFLINE`` derives
        a seed from ``global_seed``/``sample_id``/``variant``. ``ONLINE`` derives a seed from
        ``global_seed``/``epoch``/``sample_id``/``occurrence``. ``HYBRID`` deterministically
        reuses an offline variant most of the time, and derives a fresh online seed for a
        ``hybrid_online_probability`` fraction of ``(sample, epoch, occurrence)`` combinations —
        see :func:`~precisionai.agritune.augmentations.image.seeding.derive_hybrid_seed`.
    pipeline : ImageAugmentationPipeline
        The configured pipeline to apply (ignored when ``mode is AugmentationMode.NONE``).
    global_seed : int
        The run-level seed.
    variant : int, optional
        Offline variant index; ignored outside ``OFFLINE``/``HYBRID`` mode.
    epoch : int, optional
        Current training epoch; ignored outside ``ONLINE``/``HYBRID`` mode.
    occurrence : int, optional
        Occurrence-within-epoch counter; ignored outside ``ONLINE``/``HYBRID`` mode.
    hybrid_online_probability : float, optional
        Probability in ``[0, 1]`` that a given ``(sample, epoch, occurrence)`` takes the fresh
        online branch under ``HYBRID`` mode; ignored outside it.

    Returns
    -------
    PreparedSample
        The (possibly unaugmented) prepared sample.
    """
    if mode is AugmentationMode.NONE:
        return PreparedSample(sample_id=sample.sample_id, image=sample.image, target=sample.target)

    if mode is AugmentationMode.OFFLINE:
        seed = derive_offline_seed(global_seed=global_seed, sample_id=sample.sample_id, variant=variant)
    elif mode is AugmentationMode.ONLINE:
        seed = derive_online_seed(
            global_seed=global_seed, epoch=epoch, sample_id=sample.sample_id, occurrence=occurrence
        )
    elif mode is AugmentationMode.HYBRID:
        seed = derive_hybrid_seed(
            global_seed=global_seed,
            sample_id=sample.sample_id,
            variant=variant,
            epoch=epoch,
            occurrence=occurrence,
            online_probability=hybrid_online_probability,
        )
    else:  # pragma: no cover — exhaustive over AugmentationMode
        raise ValueError(f"unsupported augmentation mode: {mode}")

    return pipeline.apply(sample, seed=seed)
