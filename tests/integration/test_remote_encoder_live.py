# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Live tests against the hosted encoder API.

Skipped unless ``AGRITUNE_ENCODER_API_KEY`` is set (from the environment or a ``.env`` file).
Never logs the key. Default CI excludes this module via ``-m "not integration"``.
"""

import os

import pytest
from PIL import Image

from precisionai.agritune.encoder.gateway import EncoderGateway
from precisionai.agritune.encoder.remote import RemoteEncoderBackend, RemoteEncoderConfig
from precisionai.agritune.utils.env import ENCODER_API_KEY_VARIABLE, load_env_file

_DEFAULT_BASE_URL = "https://embeddings.precision.ai/v1"
_MODEL = "pai-embedding"


def _live_config() -> RemoteEncoderConfig:
    load_env_file()
    api_key = os.environ.get(ENCODER_API_KEY_VARIABLE) or ""
    if not api_key:
        pytest.skip(f"{ENCODER_API_KEY_VARIABLE} is not set")
    base_url = os.environ.get("AGRITUNE_ENCODER_BASE_URL", _DEFAULT_BASE_URL)
    return RemoteEncoderConfig(base_url=base_url, api_key=api_key, model=_MODEL, request_timeout_seconds=60.0)


def _image(size: tuple[int, int] = (32, 48)) -> Image.Image:
    """A non-square RGB image so native grids are allowed to be non-square."""
    return Image.new("RGB", size, color=(12, 80, 20))


@pytest.mark.integration
async def test_remote_encoder_returns_dynamic_patch_and_cls_features() -> None:
    backend = RemoteEncoderBackend(_live_config())
    features = await backend.encode([_image()])

    assert features.batch_size == 1
    assert features.encoder_model == _MODEL
    assert features.patch_tokens.ndim == 3
    _, num_patches, patch_dim = features.patch_tokens.shape
    assert num_patches >= 1
    assert patch_dim >= 1
    height, width = features.patch_grid_hw(0)
    assert height >= 1
    assert width >= 1
    assert features.cls_tokens is None or features.cls_tokens.shape == (1, features.cls_tokens.shape[1])
    if features.cls_tokens is not None:
        assert features.cls_tokens.shape[1] >= 1


@pytest.mark.integration
async def test_remote_encoder_preserves_batch_order_for_two_images() -> None:
    backend = RemoteEncoderBackend(_live_config())
    first = _image((32, 32))
    second = _image((48, 24))
    features = await EncoderGateway(backend).encode([first, second])

    assert features.batch_size == 2
    assert features.patch_tokens.shape[0] == 2
    assert features.image_sizes[0] == (32, 32)
    assert features.image_sizes[1] == (24, 48)
