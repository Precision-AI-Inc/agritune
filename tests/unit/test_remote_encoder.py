# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.encoder.remote.RemoteEncoderBackend.

No test here touches the network — ``FakeAsyncOpenAIClient`` duck-types the openai SDK's async
client, and real ``openai.*Error`` instances (backed by an in-memory ``httpx2.Request``/``Response``
— the HTTP stack this ``openai`` SDK version depends on) exercise the error-translation branches
exactly as the SDK would raise them.
"""

import base64

import httpx2
import openai
import pytest

from precisionai.agritune.encoder.errors import (
    EncoderError,
    EncoderRateLimitError,
    EncoderServerError,
    EncoderTimeoutError,
    MalformedEncoderResponseError,
)
from precisionai.agritune.encoder.remote import RemoteEncoderBackend, RemoteEncoderConfig
from tests.fixtures.fake_openai_client import (
    FakeAsyncOpenAIClient,
    FakeEmbeddingItem,
    FakeEmbeddingResponse,
    make_patch_item,
)
from tests.fixtures.image_factory import make_image

_CONFIG = RemoteEncoderConfig(base_url="https://example.test/v1", api_key="sk-pai-test", model="pai-embedding")


def _request() -> httpx2.Request:
    return httpx2.Request("POST", "https://example.test/v1/embeddings")


def _status_response(status_code: int, headers: dict[str, str] | None = None) -> httpx2.Response:
    return httpx2.Response(status_code=status_code, headers=headers or {}, request=_request())


async def test_encode_returns_features_for_single_image() -> None:
    item = make_patch_item(index=0, cls_dim=5, patch_dim=8, grid=(2, 3), seed=1)
    client = FakeAsyncOpenAIClient(response=FakeEmbeddingResponse([item]))
    backend = RemoteEncoderBackend(_CONFIG, client=client)

    features = await backend.encode([make_image((16, 16))])

    assert features.batch_size == 1
    assert features.patch_tokens.shape == (1, 6, 8)
    assert features.cls_tokens is not None
    assert features.cls_tokens.shape == (1, 5)
    assert features.patch_grid_hw(0) == (2, 3)
    assert features.encoder_model == "pai-embedding"
    assert features.encoder_revision is None
    assert features.valid_patch_mask is None


async def test_encode_records_original_image_sizes() -> None:
    item = make_patch_item(index=0, cls_dim=2, patch_dim=2, grid=(1, 1))
    client = FakeAsyncOpenAIClient(response=FakeEmbeddingResponse([item]))
    backend = RemoteEncoderBackend(_CONFIG, client=client)

    features = await backend.encode([make_image((20, 10))])  # (width, height) = (20, 10)

    assert features.image_sizes == [(10, 20)]  # (height, width)


async def test_encode_pads_variable_patch_grids_and_sets_valid_mask() -> None:
    items = [
        make_patch_item(index=0, cls_dim=2, patch_dim=4, grid=(2, 2), seed=1),  # 4 patches
        make_patch_item(index=1, cls_dim=2, patch_dim=4, grid=(2, 3), seed=2),  # 6 patches
    ]
    client = FakeAsyncOpenAIClient(response=FakeEmbeddingResponse(items))
    backend = RemoteEncoderBackend(_CONFIG, client=client)

    features = await backend.encode([make_image((8, 8)), make_image((8, 8))])

    assert features.patch_tokens.shape == (2, 6, 4)
    assert features.valid_patch_mask is not None
    assert features.valid_patch_mask[0].sum().item() == 4
    assert features.valid_patch_mask[1].sum().item() == 6


async def test_encode_sends_expected_request_shape() -> None:
    item = make_patch_item(index=0, cls_dim=2, patch_dim=2, grid=(1, 1))
    client = FakeAsyncOpenAIClient(response=FakeEmbeddingResponse([item]))
    config = RemoteEncoderConfig(
        base_url="https://example.test/v1", api_key="sk-pai-test", model="pai-embedding", native_resolution=True
    )
    backend = RemoteEncoderBackend(config, client=client)

    await backend.encode([make_image((4, 4))])

    assert client.last_call_kwargs is not None
    assert client.last_call_kwargs["model"] == "pai-embedding"
    assert client.last_call_kwargs["encoding_format"] == "base64"
    assert client.last_call_kwargs["extra_body"] == {"return_patch_tokens": True, "native_resolution": True}
    assert len(client.last_call_kwargs["input"]) == 1
    assert client.last_call_kwargs["input"][0].startswith("data:image/png;base64,")


async def test_encode_omits_native_resolution_when_unset() -> None:
    item = make_patch_item(index=0, cls_dim=2, patch_dim=2, grid=(1, 1))
    client = FakeAsyncOpenAIClient(response=FakeEmbeddingResponse([item]))
    backend = RemoteEncoderBackend(_CONFIG, client=client)

    await backend.encode([make_image((4, 4))])

    assert client.last_call_kwargs is not None
    assert client.last_call_kwargs["extra_body"] == {"return_patch_tokens": True}


async def test_empty_response_data_raises_malformed_error() -> None:
    client = FakeAsyncOpenAIClient(response=FakeEmbeddingResponse([]))
    backend = RemoteEncoderBackend(_CONFIG, client=client)
    with pytest.raises(MalformedEncoderResponseError, match="no data entries"):
        await backend.encode([make_image((4, 4))])


async def test_response_length_mismatch_raises_malformed_error() -> None:
    item = make_patch_item(index=0, cls_dim=2, patch_dim=2, grid=(1, 1))
    client = FakeAsyncOpenAIClient(response=FakeEmbeddingResponse([item]))
    backend = RemoteEncoderBackend(_CONFIG, client=client)
    with pytest.raises(MalformedEncoderResponseError, match="entries for"):
        await backend.encode([make_image((4, 4)), make_image((4, 4))])


async def test_missing_patch_fields_raises_malformed_error() -> None:
    item = FakeEmbeddingItem(index=0, embedding=[0.1, 0.2])
    client = FakeAsyncOpenAIClient(response=FakeEmbeddingResponse([item]))
    backend = RemoteEncoderBackend(_CONFIG, client=client)
    with pytest.raises(MalformedEncoderResponseError, match="return_patch_tokens"):
        await backend.encode([make_image((4, 4))])


async def test_malformed_base64_patch_embeddings_raises() -> None:
    item = FakeEmbeddingItem(
        index=0, embedding=[0.1, 0.2], patch_embeddings="not-valid-base64!!", patch_shape=[2, 1, 1]
    )
    client = FakeAsyncOpenAIClient(response=FakeEmbeddingResponse([item]))
    backend = RemoteEncoderBackend(_CONFIG, client=client)
    with pytest.raises(MalformedEncoderResponseError, match="could not decode"):
        await backend.encode([make_image((4, 4))])


async def test_empty_embedding_raises_malformed_error() -> None:
    item = FakeEmbeddingItem(
        index=0, embedding=[], patch_embeddings=base64.b64encode(b"\x00" * 4).decode(), patch_shape=[1, 1, 1]
    )
    client = FakeAsyncOpenAIClient(response=FakeEmbeddingResponse([item]))
    backend = RemoteEncoderBackend(_CONFIG, client=client)
    with pytest.raises(MalformedEncoderResponseError, match="missing 'embedding'"):
        await backend.encode([make_image((4, 4))])


async def test_rate_limit_error_is_translated_with_retry_after() -> None:
    response = _status_response(429, headers={"retry-after": "2.5"})
    exc = openai.RateLimitError("rate limited", response=response, body=None)
    client = FakeAsyncOpenAIClient(exception=exc)
    backend = RemoteEncoderBackend(_CONFIG, client=client)

    with pytest.raises(EncoderRateLimitError) as exc_info:
        await backend.encode([make_image((4, 4))])
    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after_seconds == 2.5


async def test_rate_limit_error_without_retry_after_header() -> None:
    response = _status_response(429)
    exc = openai.RateLimitError("rate limited", response=response, body=None)
    client = FakeAsyncOpenAIClient(exception=exc)
    backend = RemoteEncoderBackend(_CONFIG, client=client)

    with pytest.raises(EncoderRateLimitError) as exc_info:
        await backend.encode([make_image((4, 4))])
    assert exc_info.value.retry_after_seconds is None


async def test_rate_limit_error_with_non_numeric_retry_after_header() -> None:
    # Some servers send an HTTP-date Retry-After instead of a delta-seconds value.
    response = _status_response(429, headers={"retry-after": "Mon, 01 Jan 2026 00:00:00 GMT"})
    exc = openai.RateLimitError("rate limited", response=response, body=None)
    client = FakeAsyncOpenAIClient(exception=exc)
    backend = RemoteEncoderBackend(_CONFIG, client=client)

    with pytest.raises(EncoderRateLimitError) as exc_info:
        await backend.encode([make_image((4, 4))])
    assert exc_info.value.retry_after_seconds is None


async def test_timeout_error_is_translated() -> None:
    exc = openai.APITimeoutError(request=_request())
    client = FakeAsyncOpenAIClient(exception=exc)
    backend = RemoteEncoderBackend(_CONFIG, client=client)
    with pytest.raises(EncoderTimeoutError):
        await backend.encode([make_image((4, 4))])


async def test_internal_server_error_is_translated() -> None:
    response = _status_response(500)
    exc = openai.InternalServerError("server error", response=response, body=None)
    client = FakeAsyncOpenAIClient(exception=exc)
    backend = RemoteEncoderBackend(_CONFIG, client=client)
    with pytest.raises(EncoderServerError) as exc_info:
        await backend.encode([make_image((4, 4))])
    assert exc_info.value.status_code == 500


async def test_generic_5xx_status_error_is_translated_as_server_error() -> None:
    # A raw APIStatusError (not the InternalServerError subclass) with a 5xx status still maps
    # to EncoderServerError — defensive handling for any 5xx the SDK doesn't wrap more specifically.
    response = _status_response(503)
    exc = openai.APIStatusError("service unavailable", response=response, body=None)
    client = FakeAsyncOpenAIClient(exception=exc)
    backend = RemoteEncoderBackend(_CONFIG, client=client)
    with pytest.raises(EncoderServerError) as exc_info:
        await backend.encode([make_image((4, 4))])
    assert exc_info.value.status_code == 503


async def test_connection_error_is_translated_as_server_error() -> None:
    exc = openai.APIConnectionError(message="connection refused", request=_request())
    client = FakeAsyncOpenAIClient(exception=exc)
    backend = RemoteEncoderBackend(_CONFIG, client=client)
    with pytest.raises(EncoderServerError):
        await backend.encode([make_image((4, 4))])


async def test_generic_client_error_is_translated_as_encoder_error() -> None:
    response = _status_response(400)
    exc = openai.BadRequestError("bad request", response=response, body=None)
    client = FakeAsyncOpenAIClient(exception=exc)
    backend = RemoteEncoderBackend(_CONFIG, client=client)
    with pytest.raises(EncoderError) as exc_info:
        await backend.encode([make_image((4, 4))])
    assert exc_info.value.status_code == 400
    assert not isinstance(exc_info.value, (EncoderRateLimitError, EncoderServerError, EncoderTimeoutError))
