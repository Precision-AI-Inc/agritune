# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""The deterministic image augmentation pipeline: config, orchestration, and mode dispatch.

Supported modes: ``none`` (pass the sample through unchanged), ``offline`` (a fixed number of
precomputed variants per sample, seeded by sample + variant), and ``online`` (re-augmented every
epoch, seeded by sample + epoch + occurrence). ``hybrid`` is deferred — see
``agritune_implementation_plan.md`` §6.
"""

import random
from dataclasses import dataclass, field
from enum import Enum

from PIL import Image

from precisionai.agritune.augmentations.image import transforms
from precisionai.agritune.augmentations.image.seeding import derive_offline_seed, derive_online_seed
from precisionai.agritune.schemas.augmentation import AugmentationRecord, TransformRecord
from precisionai.agritune.schemas.samples import PreparedSample, Sample


class AugmentationMode(str, Enum):
    """Supported augmentation modes."""

    NONE = "none"
    OFFLINE = "offline"
    ONLINE = "online"


@dataclass
class GeometricConfig:
    """Geometric transform configuration. A ``None``/``0`` value disables that transform.

    Attributes
    ----------
    resize : tuple[int, int] | None
        Target ``(width, height)``, applied first if set.
    random_crop : tuple[int, int] | None
        ``(width, height)`` crop size, applied after resize if set.
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
    """Photometric transform configuration (image only). A ``None``/``0`` value disables it.

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
        Radius range sampled when blur is applied.
    noise_probability : float
        Probability in ``[0, 1]`` of applying Gaussian pixel noise.
    noise_std_range : tuple[float, float]
        Normalized-``[0, 1]``-scale standard deviation range sampled when noise is applied.
    """

    brightness_range: tuple[float, float] | None = None
    contrast_range: tuple[float, float] | None = None
    saturation_range: tuple[float, float] | None = None
    hue_max_degrees: float = 0.0
    blur_probability: float = 0.0
    blur_radius_range: tuple[float, float] = (0.1, 2.0)
    noise_probability: float = 0.0
    noise_std_range: tuple[float, float] = (0.01, 0.05)


@dataclass
class AugmentationPipelineConfig:
    """Full configuration for one :class:`ImageAugmentationPipeline`."""

    geometric: GeometricConfig = field(default_factory=GeometricConfig)
    photometric: PhotometricConfig = field(default_factory=PhotometricConfig)


class ImageAugmentationPipeline:
    """Applies a configured sequence of geometric then photometric transforms, given a seed.

    Parameters
    ----------
    config : AugmentationPipelineConfig | None, optional
        Transform configuration; defaults to every transform disabled (a no-op pipeline).
    """

    def __init__(self, config: AugmentationPipelineConfig | None = None) -> None:
        self._config = config or AugmentationPipelineConfig()

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
        rng = random.Random(seed)
        image, mask = sample.image, sample.target
        records: list[TransformRecord] = []

        image, mask, records = self._apply_geometric(image, mask, rng, records)
        image, records = self._apply_photometric(image, rng, records)

        return PreparedSample(
            sample_id=sample.sample_id,
            image=image,
            target=mask,
            augmentation_metadata=AugmentationRecord(seed=seed, transforms=records),
        )

    def _apply_geometric(
        self, image: Image.Image, mask: Image.Image, rng: random.Random, records: list[TransformRecord]
    ) -> tuple[Image.Image, Image.Image, list[TransformRecord]]:
        config = self._config.geometric
        if config.resize is not None:
            image, mask, record = transforms.resize(image, mask, config.resize)
            records.append(record)
        if config.random_crop is not None:
            image, mask, record = transforms.random_crop(image, mask, config.random_crop, rng)
            records.append(record)
        if config.horizontal_flip_probability > 0:
            image, mask, record = transforms.horizontal_flip(
                image, mask, rng, probability=config.horizontal_flip_probability
            )
            records.append(record)
        if config.vertical_flip_probability > 0:
            image, mask, record = transforms.vertical_flip(
                image, mask, rng, probability=config.vertical_flip_probability
            )
            records.append(record)
        if config.rotation_max_degrees > 0:
            image, mask, record = transforms.rotation(image, mask, config.rotation_max_degrees, rng)
            records.append(record)
        return image, mask, records

    def _apply_photometric(
        self, image: Image.Image, rng: random.Random, records: list[TransformRecord]
    ) -> tuple[Image.Image, list[TransformRecord]]:
        config = self._config.photometric
        if config.brightness_range is not None:
            image, record = transforms.brightness(image, config.brightness_range, rng)
            records.append(record)
        if config.contrast_range is not None:
            image, record = transforms.contrast(image, config.contrast_range, rng)
            records.append(record)
        if config.saturation_range is not None:
            image, record = transforms.saturation(image, config.saturation_range, rng)
            records.append(record)
        if config.hue_max_degrees > 0:
            image, record = transforms.hue(image, config.hue_max_degrees, rng)
            records.append(record)
        if config.blur_probability > 0 and rng.random() < config.blur_probability:
            image, record = transforms.blur(image, config.blur_radius_range, rng)
            records.append(record)
        if config.noise_probability > 0 and rng.random() < config.noise_probability:
            image, record = transforms.noise(image, config.noise_std_range, rng)
            records.append(record)
        return image, records


def prepare_sample(
    sample: Sample,
    *,
    mode: AugmentationMode,
    pipeline: ImageAugmentationPipeline,
    global_seed: int,
    variant: int = 0,
    epoch: int = 0,
    occurrence: int = 0,
) -> PreparedSample:
    """Dispatch a sample to the right seed derivation for ``mode`` and apply the pipeline.

    Parameters
    ----------
    sample : Sample
        The source sample.
    mode : AugmentationMode
        ``NONE`` passes the sample through with no augmentation and no seed. ``OFFLINE`` derives
        a seed from ``global_seed``/``sample_id``/``variant``. ``ONLINE`` derives a seed from
        ``global_seed``/``epoch``/``sample_id``/``occurrence``.
    pipeline : ImageAugmentationPipeline
        The configured pipeline to apply (ignored when ``mode is AugmentationMode.NONE``).
    global_seed : int
        The run-level seed.
    variant : int, optional
        Offline variant index; ignored outside ``OFFLINE`` mode.
    epoch : int, optional
        Current training epoch; ignored outside ``ONLINE`` mode.
    occurrence : int, optional
        Occurrence-within-epoch counter; ignored outside ``ONLINE`` mode.

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
    else:  # pragma: no cover — exhaustive over AugmentationMode
        raise ValueError(f"unsupported augmentation mode: {mode}")

    return pipeline.apply(sample, seed=seed)
