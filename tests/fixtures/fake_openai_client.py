# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""A fake async OpenAI-SDK-shaped client, so RemoteEncoderBackend tests never touch the network."""

import base64
from typing import Any

import numpy as np


class FakeEmbeddingItem:
    """Duck-types ``openai.types.embedding.Embedding`` plus the ``patch_*`` extension fields."""

    def __init__(
        self,
        *,
        index: int,
        embedding: list[float],
        patch_embeddings: str | None = None,
        patch_shape: list[int] | None = None,
    ) -> None:
        self.index = index
        self.embedding = embedding
        self.object = "embedding"
        if patch_embeddings is not None:
            self.patch_embeddings = patch_embeddings
        if patch_shape is not None:
            self.patch_shape = patch_shape


class FakeEmbeddingResponse:
    """Duck-types ``openai.types.create_embedding_response.CreateEmbeddingResponse``."""

    def __init__(self, data: list[FakeEmbeddingItem]) -> None:
        self.data = data
        self.model = "pai-embedding"


def make_patch_item(
    *, index: int, cls_dim: int, patch_dim: int, grid: tuple[int, int], seed: int = 0
) -> FakeEmbeddingItem:
    """Build one response item with random-but-deterministic CLS + patch tokens."""
    height, width = grid
    rng = np.random.default_rng(seed)
    embedding = rng.standard_normal(cls_dim).astype(np.float32).tolist()
    patch_array = rng.standard_normal((patch_dim, height, width)).astype("<f4")
    encoded = base64.b64encode(patch_array.tobytes()).decode("ascii")
    return FakeEmbeddingItem(
        index=index, embedding=embedding, patch_embeddings=encoded, patch_shape=[patch_dim, height, width]
    )


class _FakeEmbeddingsResource:
    def __init__(self, outer: "FakeAsyncOpenAIClient") -> None:
        self._outer = outer

    async def create(self, **kwargs: Any) -> FakeEmbeddingResponse:
        self._outer.last_call_kwargs = kwargs
        if self._outer.exception is not None:
            raise self._outer.exception
        assert self._outer.response is not None
        return self._outer.response


class FakeAsyncOpenAIClient:
    """Stands in for ``openai.AsyncOpenAI`` — exposes only ``.embeddings.create(...)``."""

    def __init__(self, *, response: FakeEmbeddingResponse | None = None, exception: Exception | None = None) -> None:
        self.response = response
        self.exception = exception
        self.last_call_kwargs: dict[str, Any] | None = None
        self.embeddings = _FakeEmbeddingsResource(self)
