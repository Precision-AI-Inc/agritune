# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""``FakeEncoderBackend`` — the encoder every unit test and CI run exercises.

No unit test in this repository requires a live encoder API key. Configurable patch dimension,
CLS dimension, patch grid, latency, and injectable failure modes let the rest of the codebase
(gateway, retry policy, feature cache, training loop) be tested against realistic encoder
behavior — including the failure modes the real hosted API can produce — without any network
access. See ``docs/encoder.md`` and CLAUDE.md's "Testing against the encoder" section.
"""

import asyncio
import random
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from precisionai.agritune.encoder.base import ImageInput
from precisionai.agritune.encoder.errors import (
    EncoderRateLimitError,
    EncoderServerError,
    EncoderTimeoutError,
    MalformedEncoderResponseError,
)
from precisionai.agritune.logging import get_logger
from precisionai.agritune.schemas.features import EncoderFeatures

logger = get_logger(__name__)


@dataclass
class FakeEncoderConfig:
    """Configuration for :class:`FakeEncoderBackend`.

    Attributes
    ----------
    patch_dim : int
        Patch embedding dimension ``D``.
    cls_dim : int | None
        CLS embedding dimension ``D_cls``, or ``None`` to simulate an encoder/model with no CLS
        token.
    patch_grid : tuple[int, int]
        Fixed ``(H, W)`` patch grid produced for every image.
    encoder_model : str
        Value reported as ``EncoderFeatures.encoder_model``.
    encoder_revision : str | None
        Value reported as ``EncoderFeatures.encoder_revision``.
    latency_seconds : float
        Simulated per-request latency, applied via ``asyncio.sleep`` before any failure check.
    failure_probability : float
        Probability in ``[0, 1]`` that a call raises :class:`EncoderRateLimitError` (when
        ``status_code_on_failure == 429``) or :class:`EncoderServerError` (otherwise).
    status_code_on_failure : int
        HTTP status code to simulate when a failure is triggered by ``failure_probability``.
    timeout_probability : float
        Probability in ``[0, 1]`` that a call raises :class:`EncoderTimeoutError`.
    malformed_response_probability : float
        Probability in ``[0, 1]`` that a call raises :class:`MalformedEncoderResponseError`.
    seed : int
        Seeds both the failure-injection RNG and the deterministic feature generator.
    """

    patch_dim: int = 384
    cls_dim: int | None = 384
    patch_grid: tuple[int, int] = (14, 14)
    encoder_model: str = "fake-encoder"
    encoder_revision: str | None = "fake-v1"
    latency_seconds: float = 0.0
    failure_probability: float = 0.0
    status_code_on_failure: int = 500
    timeout_probability: float = 0.0
    malformed_response_probability: float = 0.0
    seed: int = 0


class FakeEncoderBackend:
    """An :class:`~precisionai.agritune.schemas.protocols.EncoderBackend` with no network access.

    Parameters
    ----------
    config : FakeEncoderConfig | None, optional
        Behavior configuration; defaults to :class:`FakeEncoderConfig` with no simulated latency
        or failures.
    """

    def __init__(self, config: FakeEncoderConfig | None = None) -> None:
        self._config = config or FakeEncoderConfig()
        self._failure_rng = random.Random(self._config.seed)
        self._call_count = 0

    @property
    def call_count(self) -> int:
        """Return the number of times :meth:`encode` has been called."""
        return self._call_count

    async def encode(self, images: Sequence[ImageInput]) -> EncoderFeatures:
        """Return deterministic fake features, or raise an injected failure.

        Parameters
        ----------
        images : Sequence[ImageInput]
            Images to "encode" — content is never inspected.

        Returns
        -------
        EncoderFeatures
            Deterministic features shaped by :attr:`FakeEncoderConfig`.

        Raises
        ------
        EncoderTimeoutError
            When ``timeout_probability`` triggers.
        EncoderRateLimitError
            When ``failure_probability`` triggers with ``status_code_on_failure == 429``.
        EncoderServerError
            When ``failure_probability`` triggers with any other ``status_code_on_failure``.
        MalformedEncoderResponseError
            When ``malformed_response_probability`` triggers.
        """
        self._call_count += 1
        config = self._config
        logger.debug("fake-encoding %d image(s) (call #%d)", len(images), self._call_count)

        if config.latency_seconds > 0:
            await asyncio.sleep(config.latency_seconds)

        if config.timeout_probability > 0 and self._failure_rng.random() < config.timeout_probability:
            raise EncoderTimeoutError("FakeEncoderBackend: simulated timeout")

        if config.failure_probability > 0 and self._failure_rng.random() < config.failure_probability:
            status_code = config.status_code_on_failure
            if status_code == 429:
                raise EncoderRateLimitError(
                    "FakeEncoderBackend: simulated 429", status_code=429, retry_after_seconds=1.0
                )
            raise EncoderServerError(f"FakeEncoderBackend: simulated {status_code}", status_code=status_code)

        if (
            config.malformed_response_probability > 0
            and self._failure_rng.random() < config.malformed_response_probability
        ):
            raise MalformedEncoderResponseError("FakeEncoderBackend: simulated malformed response")

        return self._build_features(batch_size=len(images))

    def _build_features(self, *, batch_size: int) -> EncoderFeatures:
        height, width = self._config.patch_grid
        num_patches = height * width
        generator = torch.Generator().manual_seed(self._config.seed + self._call_count)

        patch_tokens = torch.randn(batch_size, num_patches, self._config.patch_dim, generator=generator)
        cls_tokens = (
            torch.randn(batch_size, self._config.cls_dim, generator=generator)
            if self._config.cls_dim is not None
            else None
        )
        return EncoderFeatures(
            patch_tokens=patch_tokens,
            cls_tokens=cls_tokens,
            patch_grid=torch.tensor([[height, width]] * batch_size, dtype=torch.long),
            valid_patch_mask=None,
            image_sizes=[(224, 224)] * batch_size,
            encoder_model=self._config.encoder_model,
            encoder_revision=self._config.encoder_revision,
        )
