# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Feature cache key derivation.

A cached feature must be invalidated whenever anything that affects it changes: sample ID, image
content, augmentation configuration/seed, encoder model/revision/preprocessing, or the feature
schema version itself. See ``docs/feature-caching.md``.
"""

import hashlib
import json
from dataclasses import dataclass

from precisionai.agritune.schemas.augmentation import AugmentationRecord

FEATURE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class EncoderFingerprint:
    """Identifies the encoder configuration that produced (or will produce) a feature.

    Attributes
    ----------
    model : str
        Encoder model identifier/alias.
    revision : str | None
        Encoder revision, when known — see ``docs/encoder.md`` on why this is often ``None``.
    preprocessing : str
        A stable string representation of preprocessing parameters (e.g. resize size,
        normalization, ``native_resolution`` flag) that affect the features produced.
    """

    model: str
    revision: str | None
    preprocessing: str


def hash_image_bytes(data: bytes) -> str:
    """Return a stable hex digest of raw image bytes.

    Parameters
    ----------
    data : bytes
        Raw encoded image bytes (e.g. the contents of a PNG/JPEG file).

    Returns
    -------
    str
        A SHA-256 hex digest.
    """
    return hashlib.sha256(data).hexdigest()


def hash_augmentation(record: AugmentationRecord | None) -> str:
    """Return a stable fingerprint of an augmentation record.

    Parameters
    ----------
    record : AugmentationRecord | None
        The augmentation applied to a sample, or ``None`` when ``augmentation.mode: none``.

    Returns
    -------
    str
        ``"none"`` when ``record`` is ``None``; otherwise a SHA-256 hex digest of the seed and
        every transform's name and parameters, in order.
    """
    if record is None:
        return "none"
    payload = {
        "seed": record.seed,
        "transforms": [{"name": transform.name, "params": transform.params} for transform in record.transforms],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def compute_feature_key(
    *,
    sample_id: str,
    image_hash: str,
    augmentation_fingerprint: str,
    encoder_fingerprint: EncoderFingerprint,
    feature_schema_version: int = FEATURE_SCHEMA_VERSION,
) -> str:
    """Compute the cache key for one sample's features.

    Parameters
    ----------
    sample_id : str
        The sample's stable identifier.
    image_hash : str
        Output of :func:`hash_image_bytes` for this sample's image.
    augmentation_fingerprint : str
        Output of :func:`hash_augmentation` for this sample's applied augmentation.
    encoder_fingerprint : EncoderFingerprint
        Identifies the encoder configuration.
    feature_schema_version : int, optional
        Bump this whenever :class:`~precisionai.agritune.schemas.features.EncoderFeatures`'s
        on-disk representation changes incompatibly, to invalidate all prior caches at once.

    Returns
    -------
    str
        A SHA-256 hex digest stable across processes and runs, changing whenever any input here
        changes.
    """
    payload = {
        "sample_id": sample_id,
        "image_hash": image_hash,
        "augmentation_fingerprint": augmentation_fingerprint,
        "encoder_model": encoder_fingerprint.model,
        "encoder_revision": encoder_fingerprint.revision or "",
        "encoder_preprocessing": encoder_fingerprint.preprocessing,
        "feature_schema_version": feature_schema_version,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()
