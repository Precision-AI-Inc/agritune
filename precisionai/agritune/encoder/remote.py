# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""``RemoteEncoderBackend`` — wraps the hosted, OpenAI-SDK-compatible embeddings API.

See ``docs/encoder.md`` for the full API contract this module implements: the CLS token comes
back as ``embedding``, patch tokens come back as ``patch_embeddings`` (channels-first
``(D, H, W)``, base64 little-endian float32) plus ``patch_shape``, requested via
``extra_body={"return_patch_tokens": True}``. Only ``PIL.Image.Image`` inputs are supported —
matching ``precisionai.agritune.data.dataset.ManifestDataset``'s output.

This backend has no training-specific retry/backoff behavior — that belongs to
``precisionai.agritune.encoder.gateway.EncoderGateway``. It only translates SDK/HTTP errors into
the ``precisionai.agritune.encoder.errors`` hierarchy.
"""

import base64
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
import openai
import torch
from PIL import Image

from precisionai.agritune.encoder.errors import (
    EncoderError,
    EncoderRateLimitError,
    EncoderServerError,
    EncoderTimeoutError,
    MalformedEncoderResponseError,
)
from precisionai.agritune.logging import get_logger
from precisionai.agritune.schemas.features import EncoderFeatures

logger = get_logger(__name__)


@dataclass
class RemoteEncoderConfig:
    """Configuration for :class:`RemoteEncoderBackend`.

    Attributes
    ----------
    base_url : str
        Base URL of the hosted embeddings deployment (e.g. ``"https://.../v1"``).
    api_key : str
        Bearer credential (``sk-pai-...`` API key or a Cognito access token). Never logged.
    model : str
        Model identifier/alias to request (default: the public ``"pai-embedding"`` alias).
    native_resolution : bool | None
        When set, forwarded as the ``native_resolution`` request extension; ``None`` omits it
        (server default applies).
    request_timeout_seconds : float
        Client-side request timeout. The server enforces no timeout of its own — see
        ``docs/encoder.md``.
    """

    base_url: str
    api_key: str
    model: str = "pai-embedding"
    native_resolution: bool | None = None
    request_timeout_seconds: float = 60.0


def _image_to_data_uri(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _decode_cls_embedding(embedding: Any) -> torch.Tensor:
    """Decode a CLS vector from a float list or a little-endian float32 base64 buffer."""
    if isinstance(embedding, str):
        try:
            decoded = np.frombuffer(base64.b64decode(embedding, validate=True), dtype="<f4")
        except (ValueError, TypeError) as exc:
            raise MalformedEncoderResponseError(f"could not decode embedding: {exc}") from exc
        if decoded.size < 1:
            raise MalformedEncoderResponseError("encoder response item missing 'embedding'")
        return torch.from_numpy(np.array(decoded, copy=True)).to(dtype=torch.float32)
    try:
        vector = torch.as_tensor(embedding, dtype=torch.float32).reshape(-1)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise MalformedEncoderResponseError(f"could not decode embedding: {exc}") from exc
    if vector.numel() < 1:
        raise MalformedEncoderResponseError("encoder response item missing 'embedding'")
    return vector.contiguous()


def _retry_after_seconds(exc: openai.APIStatusError) -> float | None:
    header = exc.response.headers.get("retry-after")
    if header is None:
        return None
    try:
        return float(header)
    except ValueError:
        return None


class RemoteEncoderBackend:
    """An :class:`~precisionai.agritune.schemas.protocols.EncoderBackend` over the hosted API.

    Parameters
    ----------
    config : RemoteEncoderConfig
        Connection and request configuration.
    client : Any | None, optional
        An existing async client to reuse — normally an ``openai.AsyncOpenAI``, but any object
        exposing an awaitable ``.embeddings.create(...)`` with the same shape works (this is the
        seam unit tests inject a fake client through). A new ``openai.AsyncOpenAI`` is constructed
        from ``config`` when omitted.
    """

    def __init__(self, config: RemoteEncoderConfig, *, client: Any | None = None) -> None:
        self._config = config
        self._client = client or openai.AsyncOpenAI(
            base_url=config.base_url, api_key=config.api_key, timeout=config.request_timeout_seconds
        )

    async def encode(self, images: Sequence[Image.Image]) -> EncoderFeatures:
        """Encode a batch of images via the hosted embeddings API.

        Parameters
        ----------
        images : Sequence[PIL.Image.Image]
            Images to encode.

        Returns
        -------
        EncoderFeatures
            Batched features, one entry per input image, in the same order.

        Raises
        ------
        EncoderRateLimitError
            On HTTP 429 (quota exceeded); carries ``retry_after_seconds`` when the server sent one.
        EncoderTimeoutError
            When the client-side timeout elapses, or the connection fails outright.
        EncoderServerError
            On HTTP 5xx.
        EncoderError
            On any other HTTP error status (e.g. 400/401/403/404/422).
        MalformedEncoderResponseError
            When the response is missing expected fields or cannot be decoded.
        """
        image_sizes = [(image.height, image.width) for image in images]
        payload = [_image_to_data_uri(image) for image in images]

        extra_body: dict[str, Any] = {"return_patch_tokens": True}
        if self._config.native_resolution is not None:
            extra_body["native_resolution"] = self._config.native_resolution

        # Never log the payload/headers themselves (data URIs, Authorization) — only shapes/counts.
        logger.debug("requesting model=%s for %d image(s), sizes=%s", self._config.model, len(images), image_sizes)

        try:
            response = await self._client.embeddings.create(
                model=self._config.model,
                input=payload,
                encoding_format="base64",
                extra_body=extra_body,
            )
        except openai.RateLimitError as exc:
            raise EncoderRateLimitError(
                str(exc), status_code=429, retry_after_seconds=_retry_after_seconds(exc)
            ) from exc
        except openai.APITimeoutError as exc:
            raise EncoderTimeoutError(str(exc)) from exc
        except openai.InternalServerError as exc:
            raise EncoderServerError(str(exc), status_code=exc.status_code) from exc
        except openai.APIConnectionError as exc:
            raise EncoderServerError(str(exc)) from exc
        except openai.APIStatusError as exc:
            if exc.status_code >= 500:
                raise EncoderServerError(str(exc), status_code=exc.status_code) from exc
            raise EncoderError(str(exc), status_code=exc.status_code) from exc

        return self._parse_response(response, image_sizes=image_sizes, model=self._config.model)

    def _parse_response(self, response: Any, *, image_sizes: list[tuple[int, int]], model: str) -> EncoderFeatures:
        if not response.data:
            raise MalformedEncoderResponseError("encoder response contained no data entries")
        if len(response.data) != len(image_sizes):
            raise MalformedEncoderResponseError(
                f"encoder returned {len(response.data)} entries for {len(image_sizes)} images"
            )

        try:
            raw_indices = [item.index for item in response.data]
        except AttributeError as exc:
            raise MalformedEncoderResponseError("encoder response item has an invalid or missing index") from exc
        if any(isinstance(index, bool) or not isinstance(index, int) for index in raw_indices):
            raise MalformedEncoderResponseError("encoder response item has an invalid or missing index")
        indices = raw_indices
        expected_indices = list(range(len(image_sizes)))
        if sorted(indices) != expected_indices:
            raise MalformedEncoderResponseError(
                f"encoder response indices must be exactly {expected_indices}; got {indices}"
            )

        cls_vectors: list[torch.Tensor] = []
        patch_token_list: list[torch.Tensor] = []
        grids: list[tuple[int, int]] = []

        for item in sorted(response.data, key=lambda entry: int(entry.index)):
            if not item.embedding:
                raise MalformedEncoderResponseError("encoder response item missing 'embedding'")
            cls_vectors.append(_decode_cls_embedding(item.embedding))

            patch_tokens, grid = self._decode_patch_item(item)
            patch_token_list.append(patch_tokens)
            grids.append(grid)

        return self._build_features(cls_vectors, patch_token_list, grids, image_sizes=image_sizes, model=model)

    @staticmethod
    def _decode_patch_item(item: Any) -> tuple[torch.Tensor, tuple[int, int]]:
        patch_embeddings = getattr(item, "patch_embeddings", None)
        patch_shape = getattr(item, "patch_shape", None)
        if patch_embeddings is None or patch_shape is None:
            raise MalformedEncoderResponseError(
                "encoder response item missing 'patch_embeddings'/'patch_shape' — was return_patch_tokens requested?"
            )
        try:
            if any(isinstance(dim, bool) or not isinstance(dim, int) for dim in patch_shape):
                raise ValueError(f"patch_shape dimensions must be integers; got {patch_shape}")
            patch_dim, height, width = patch_shape
            if patch_dim <= 0 or height <= 0 or width <= 0:
                raise ValueError(f"patch_shape dimensions must be positive; got {patch_shape}")
            decoded = base64.b64decode(patch_embeddings, validate=True)
            array = np.frombuffer(decoded, dtype="<f4").reshape(patch_dim, height, width)
        except (ValueError, TypeError) as exc:
            raise MalformedEncoderResponseError(f"could not decode patch_embeddings: {exc}") from exc

        # (D, H, W) channels-first -> (N, D) patch-tokens-last, N = H * W.
        patch_tokens = torch.from_numpy(array.copy()).reshape(patch_dim, height * width).T.contiguous()
        return patch_tokens, (height, width)

    @staticmethod
    def _build_features(
        cls_vectors: list[torch.Tensor],
        patch_token_list: list[torch.Tensor],
        grids: list[tuple[int, int]],
        *,
        image_sizes: list[tuple[int, int]],
        model: str,
    ) -> EncoderFeatures:
        batch_size = len(patch_token_list)
        max_patches = max(tokens.shape[0] for tokens in patch_token_list)
        patch_dim = patch_token_list[0].shape[1]
        if any(tokens.shape[1] != patch_dim for tokens in patch_token_list):
            raise MalformedEncoderResponseError("encoder response items have inconsistent patch dimensions")
        cls_dim = cls_vectors[0].shape[0]
        if any(vector.shape[0] != cls_dim for vector in cls_vectors):
            raise MalformedEncoderResponseError("encoder response items have inconsistent CLS dimensions")

        padded = torch.zeros(batch_size, max_patches, patch_dim, dtype=torch.float32)
        valid_mask = torch.zeros(batch_size, max_patches, dtype=torch.bool)
        for index, tokens in enumerate(patch_token_list):
            count = tokens.shape[0]
            padded[index, :count] = tokens
            valid_mask[index, :count] = True
        needs_mask = any(tokens.shape[0] != max_patches for tokens in patch_token_list)

        return EncoderFeatures(
            patch_tokens=padded,
            cls_tokens=torch.stack(cls_vectors),
            patch_grid=torch.tensor(grids, dtype=torch.long),
            valid_patch_mask=valid_mask if needs_mask else None,
            image_sizes=image_sizes,
            encoder_model=model,
            encoder_revision=None,
        )
