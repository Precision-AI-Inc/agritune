# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.services.encoder_selection.

Shared by the CLI and API layer — see ``precisionai.agritune.cli.handlers`` and
``precisionai.agritune.api.routes`` for the two callers.
"""

from precisionai.agritune.encoder.fake import FakeEncoderBackend
from precisionai.agritune.encoder.gateway import EncoderGateway
from precisionai.agritune.encoder.remote import RemoteEncoderBackend
from precisionai.agritune.services.encoder_selection import build_encoder, build_raw_encoder


def test_build_raw_encoder_selects_fake_backend_by_default() -> None:
    backend, fingerprint = build_raw_encoder(base_url=None, api_key=None, model="pai-embedding", preprocessing="")
    assert isinstance(backend, FakeEncoderBackend)
    assert fingerprint.model == "fake-encoder"
    assert fingerprint.revision == "fake-v1"


def test_build_raw_encoder_selects_remote_backend_when_base_url_given() -> None:
    backend, fingerprint = build_raw_encoder(
        base_url="https://example.test/v1", api_key="sk-test", model="pai-embedding", preprocessing=""
    )
    assert isinstance(backend, RemoteEncoderBackend)
    assert fingerprint.model == "pai-embedding"
    assert fingerprint.revision is None


def test_build_raw_encoder_preserves_preprocessing_label() -> None:
    _, fingerprint = build_raw_encoder(base_url=None, api_key=None, model="pai-embedding", preprocessing="resize=224")
    assert fingerprint.preprocessing == "resize=224"


def test_build_encoder_wraps_the_raw_backend_in_a_gateway() -> None:
    gateway, fingerprint = build_encoder(base_url=None, api_key=None, model="pai-embedding", preprocessing="")
    assert isinstance(gateway, EncoderGateway)
    assert fingerprint.model == "fake-encoder"
