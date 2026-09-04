# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Encoder backend selection shared by the CLI and the API layer.

Both entry points need to turn a handful of plain arguments (an optional hosted-API base URL, an
API key, a model alias, a preprocessing label) into a concrete
:class:`~precisionai.agritune.schemas.protocols.EncoderBackend` and its
:class:`~precisionai.agritune.features.keys.EncoderFingerprint` — kept here once so neither
duplicates the other, per CLAUDE.md's "CLI ─┐ / API ─┼──> Services" layering.
"""

from precisionai.agritune.encoder.fake import FakeEncoderBackend
from precisionai.agritune.encoder.gateway import EncoderGateway
from precisionai.agritune.encoder.remote import RemoteEncoderBackend, RemoteEncoderConfig
from precisionai.agritune.features.keys import EncoderFingerprint
from precisionai.agritune.logging import get_logger, redact_text
from precisionai.agritune.schemas.protocols import EncoderBackend

logger = get_logger(__name__)


def build_raw_encoder(
    *, base_url: str | None, api_key: str | None, model: str, preprocessing: str
) -> tuple[EncoderBackend, EncoderFingerprint]:
    """Select a raw (non-gateway-wrapped) encoder backend and its fingerprint.

    Parameters
    ----------
    base_url : str | None
        Hosted encoder API base URL; ``None``/empty selects :class:`FakeEncoderBackend` instead.
    api_key : str | None
        Encoder API key; ignored when ``base_url`` is not given.
    model : str
        Encoder model alias.
    preprocessing : str
        A stable label for preprocessing params, for cache invalidation.

    Returns
    -------
    tuple[EncoderBackend, EncoderFingerprint]
    """
    if base_url:
        backend: EncoderBackend = RemoteEncoderBackend(
            RemoteEncoderConfig(base_url=base_url, api_key=api_key or "", model=model)
        )
        fingerprint = EncoderFingerprint(model=model, revision=None, preprocessing=preprocessing)
        logger.info("using RemoteEncoderBackend: base_url=%s model=%s", redact_text(base_url), model)
    else:
        backend = FakeEncoderBackend()
        fingerprint = EncoderFingerprint(model="fake-encoder", revision="fake-v1", preprocessing=preprocessing)
        logger.info("using FakeEncoderBackend (no base_url configured)")
    return backend, fingerprint


def build_encoder(
    *, base_url: str | None, api_key: str | None, model: str, preprocessing: str
) -> tuple[EncoderGateway, EncoderFingerprint]:
    """Select an encoder backend and wrap it in an :class:`EncoderGateway`.

    Parameters mirror :func:`build_raw_encoder`.

    Returns
    -------
    tuple[EncoderGateway, EncoderFingerprint]
    """
    backend, fingerprint = build_raw_encoder(
        base_url=base_url, api_key=api_key, model=model, preprocessing=preprocessing
    )
    return EncoderGateway(backend), fingerprint
