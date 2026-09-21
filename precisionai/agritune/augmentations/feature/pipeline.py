# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Composes feature-space transforms into one configured, order-fixed pipeline.

Mirrors :class:`~precisionai.agritune.augmentations.image.pipeline.ImageAugmentationPipeline`'s
config-driven, "``0``/``None`` disables it" shape, but for
:class:`~precisionai.agritune.schemas.features.EncoderFeatures` instead of images.
"""

from dataclasses import dataclass

import torch

from precisionai.agritune.augmentations.feature.transforms import (
    cls_dropout,
    feature_channel_dropout,
    gaussian_feature_noise,
    patch_dropout,
    token_masking,
)
from precisionai.agritune.schemas.features import EncoderFeatures


@dataclass
class FeatureAugmentationConfig:
    """Feature-space augmentation configuration. A ``0.0`` probability/std disables that transform.

    Attributes
    ----------
    patch_dropout_probability : float
        Drop probability per patch, in ``[0, 1)``.
    token_masking_probability : float
        Mask probability per patch, in ``[0, 1]``.
    token_mask_value : float
        Constant value masked patches are set to.
    gaussian_noise_std : float
        Standard deviation of additive Gaussian noise on patch/CLS tokens.
    cls_dropout_probability : float
        Drop probability per sample's CLS token, in ``[0, 1)``; a no-op when a sample has none.
    channel_dropout_probability : float
        Drop probability per embedding channel, in ``[0, 1)``.
    """

    patch_dropout_probability: float = 0.0
    token_masking_probability: float = 0.0
    token_mask_value: float = 0.0
    gaussian_noise_std: float = 0.0
    cls_dropout_probability: float = 0.0
    channel_dropout_probability: float = 0.0


class FeatureAugmentationPipeline:
    """Applies a configured sequence of feature-space transforms to a batch of features.

    Order is fixed: patch dropout, token masking, Gaussian noise, CLS dropout, then channel
    dropout — each only applied when its config value is non-zero.

    Parameters
    ----------
    config : FeatureAugmentationConfig | None, optional
        Transform configuration; defaults to every transform disabled (a no-op pipeline).
    """

    def __init__(self, config: FeatureAugmentationConfig | None = None) -> None:
        self._config = config or FeatureAugmentationConfig()

    def apply(self, features: EncoderFeatures, *, generator: torch.Generator | None = None) -> EncoderFeatures:
        """Apply every enabled transform to ``features``, in order.

        Parameters
        ----------
        features : EncoderFeatures
            Features to augment (typically straight from a ``FeatureProvider``).
        generator : torch.Generator | None, optional
            RNG source shared across every enabled transform; ``None`` uses PyTorch's global RNG,
            so the perturbation is captured by the trainer's full RNG-state checkpointing.

        Returns
        -------
        EncoderFeatures
            The augmented features. Identical to the input when every transform is disabled.
        """
        config = self._config
        if config.patch_dropout_probability > 0:
            features = patch_dropout(features, probability=config.patch_dropout_probability, generator=generator)
        if config.token_masking_probability > 0:
            features = token_masking(
                features,
                probability=config.token_masking_probability,
                mask_value=config.token_mask_value,
                generator=generator,
            )
        if config.gaussian_noise_std > 0:
            features = gaussian_feature_noise(features, std=config.gaussian_noise_std, generator=generator)
        if config.cls_dropout_probability > 0:
            features = cls_dropout(features, probability=config.cls_dropout_probability, generator=generator)
        if config.channel_dropout_probability > 0:
            features = feature_channel_dropout(
                features, probability=config.channel_dropout_probability, generator=generator
            )
        return features
