# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Feature-space augmentation transforms — operate on cached/online :class:`EncoderFeatures`.

Unlike image augmentation, these require no extra encoder API calls: they perturb patch/CLS
tokens already returned by a ``FeatureProvider``, right before a decoder consumes them. Every
transform accepts an optional ``torch.Generator``; when omitted, it draws from PyTorch's global
RNG, so the perturbation is captured by the same full RNG-state checkpointing every other
stochastic training operation already relies on (see
:mod:`precisionai.agritune.training.checkpointing`) — no separate seed-derivation scheme is
needed here.

``patch_dropout``/``feature_channel_dropout``/``cls_dropout`` rescale surviving values by
``1 / (1 - probability)`` (inverted dropout, like ``torch.nn.Dropout``), so a decoder trained with
these enabled sees consistent expected magnitudes whether or not they are active at eval time.
``token_masking`` does not rescale — it targets masked-token-style objectives, where the model is
meant to notice a token was replaced, not compensate for it.
"""

import torch

from precisionai.agritune.schemas.features import EncoderFeatures


def _validate_dropout_probability(probability: float) -> None:
    if not 0.0 <= probability < 1.0:
        raise ValueError(f"probability must be in [0, 1); got {probability}")


def patch_dropout(
    features: EncoderFeatures, *, probability: float, generator: torch.Generator | None = None
) -> EncoderFeatures:
    """Zero whole patch embeddings independently with ``probability``, rescaling survivors.

    Parameters
    ----------
    features : EncoderFeatures
    probability : float
        Drop probability per patch, in ``[0, 1)``.
    generator : torch.Generator | None, optional
        RNG source; ``None`` uses PyTorch's global RNG.

    Returns
    -------
    EncoderFeatures
        A copy with ``patch_tokens`` modified; every other field unchanged.
    """
    _validate_dropout_probability(probability)
    if probability == 0.0:
        return features

    batch_size, num_patches, _ = features.patch_tokens.shape
    # torch.rand always draws on the generator's own device (CPU, for the shared global/checkpointed
    # RNG this defaults to) — moved to match patch_tokens explicitly, since that may be a CUDA
    # device once Trainer starts moving batches there.
    keep = (torch.rand(batch_size, num_patches, generator=generator) >= probability).to(features.patch_tokens.device)
    scaled = features.patch_tokens * keep.unsqueeze(-1) / (1.0 - probability)
    return _replace(features, patch_tokens=scaled)


def token_masking(
    features: EncoderFeatures,
    *,
    probability: float,
    mask_value: float = 0.0,
    generator: torch.Generator | None = None,
) -> EncoderFeatures:
    """Replace whole patch embeddings with a constant mask value, independently, with ``probability``.

    Parameters
    ----------
    features : EncoderFeatures
    probability : float
        Mask probability per patch, in ``[0, 1]``.
    mask_value : float, optional
        Constant value every masked patch's embedding is set to.
    generator : torch.Generator | None, optional
        RNG source; ``None`` uses PyTorch's global RNG.

    Returns
    -------
    EncoderFeatures
        A copy with ``patch_tokens`` modified; every other field unchanged.

    Raises
    ------
    ValueError
        If ``probability`` is outside ``[0, 1]``.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1]; got {probability}")
    if probability == 0.0:
        return features

    batch_size, num_patches, _ = features.patch_tokens.shape
    mask = (torch.rand(batch_size, num_patches, generator=generator) < probability).to(features.patch_tokens.device)
    masked = features.patch_tokens.clone()
    masked[mask] = mask_value
    return _replace(features, patch_tokens=masked)


def gaussian_feature_noise(
    features: EncoderFeatures, *, std: float, generator: torch.Generator | None = None
) -> EncoderFeatures:
    """Add zero-mean Gaussian noise to patch tokens (and CLS tokens, when present).

    Parameters
    ----------
    features : EncoderFeatures
    std : float
        Standard deviation of the additive noise; must be ``>= 0``.
    generator : torch.Generator | None, optional
        RNG source; ``None`` uses PyTorch's global RNG.

    Returns
    -------
    EncoderFeatures
        A copy with ``patch_tokens`` (and ``cls_tokens``, if present) perturbed.

    Raises
    ------
    ValueError
        If ``std`` is negative.
    """
    if std < 0:
        raise ValueError(f"std must be >= 0; got {std}")
    if std == 0.0:
        return features

    patch_noise = torch.randn(features.patch_tokens.shape, generator=generator).to(features.patch_tokens.device)
    noisy_patch = features.patch_tokens + patch_noise * std
    noisy_cls = features.cls_tokens
    if features.cls_tokens is not None:
        cls_noise = torch.randn(features.cls_tokens.shape, generator=generator).to(features.cls_tokens.device)
        noisy_cls = features.cls_tokens + cls_noise * std
    return _replace(features, patch_tokens=noisy_patch, cls_tokens=noisy_cls)


def cls_dropout(
    features: EncoderFeatures, *, probability: float, generator: torch.Generator | None = None
) -> EncoderFeatures:
    """Zero a whole sample's CLS embedding with ``probability``, rescaling survivors.

    A no-op when ``features.cls_tokens`` is ``None`` — not every encoder/model produces one.

    Parameters
    ----------
    features : EncoderFeatures
    probability : float
        Drop probability per sample, in ``[0, 1)``.
    generator : torch.Generator | None, optional
        RNG source; ``None`` uses PyTorch's global RNG.

    Returns
    -------
    EncoderFeatures
        A copy with ``cls_tokens`` modified; every other field unchanged.
    """
    _validate_dropout_probability(probability)
    if features.cls_tokens is None or probability == 0.0:
        return features

    batch_size = features.cls_tokens.shape[0]
    keep = (torch.rand(batch_size, generator=generator) >= probability).to(features.cls_tokens.device)
    scaled = features.cls_tokens * keep.unsqueeze(-1) / (1.0 - probability)
    return _replace(features, cls_tokens=scaled)


def feature_channel_dropout(
    features: EncoderFeatures, *, probability: float, generator: torch.Generator | None = None
) -> EncoderFeatures:
    """Zero whole embedding channels, shared across every patch/sample, rescaling survivors.

    ``patch_tokens`` and ``cls_tokens`` (when present) get independently drawn channel masks,
    since ``D`` and ``D_cls`` need not match.

    Parameters
    ----------
    features : EncoderFeatures
    probability : float
        Drop probability per channel, in ``[0, 1)``.
    generator : torch.Generator | None, optional
        RNG source; ``None`` uses PyTorch's global RNG.

    Returns
    -------
    EncoderFeatures
        A copy with ``patch_tokens`` (and ``cls_tokens``, if present) modified.
    """
    _validate_dropout_probability(probability)
    if probability == 0.0:
        return features

    patch_dim = features.patch_tokens.shape[-1]
    keep_patch = (torch.rand(patch_dim, generator=generator) >= probability).to(
        device=features.patch_tokens.device, dtype=features.patch_tokens.dtype
    )
    scaled_patch = features.patch_tokens * keep_patch / (1.0 - probability)

    scaled_cls = features.cls_tokens
    if features.cls_tokens is not None:
        cls_dim = features.cls_tokens.shape[-1]
        keep_cls = (torch.rand(cls_dim, generator=generator) >= probability).to(
            device=features.cls_tokens.device, dtype=features.cls_tokens.dtype
        )
        scaled_cls = features.cls_tokens * keep_cls / (1.0 - probability)

    return _replace(features, patch_tokens=scaled_patch, cls_tokens=scaled_cls)


def _replace(
    features: EncoderFeatures, *, patch_tokens: torch.Tensor | None = None, cls_tokens: torch.Tensor | None = None
) -> EncoderFeatures:
    return EncoderFeatures(
        patch_tokens=patch_tokens if patch_tokens is not None else features.patch_tokens,
        cls_tokens=cls_tokens if cls_tokens is not None else features.cls_tokens,
        patch_grid=features.patch_grid,
        valid_patch_mask=features.valid_patch_mask,
        image_sizes=features.image_sizes,
        encoder_model=features.encoder_model,
        encoder_revision=features.encoder_revision,
        metadata=features.metadata,
    )
