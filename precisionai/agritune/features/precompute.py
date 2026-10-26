# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Resumable offline feature precomputation — powers ``agritune features build``.

If N of M samples were already encoded (i.e. already present in the target store under their
computed cache key), restarting only encodes the missing ``M - N``. A sample already present is
never re-encoded, even under a different process, host, or interrupted run.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from precisionai.agritune.encoder.errors import EncoderError
from precisionai.agritune.features.keys import EncoderFingerprint, compute_feature_key, hash_augmentation
from precisionai.agritune.logging import get_logger
from precisionai.agritune.schemas.features import EncoderFeatures, select_one
from precisionai.agritune.schemas.protocols import EncoderBackend
from precisionai.agritune.schemas.samples import PreparedSample

logger = get_logger(__name__)

# Uncached samples sent to the encoder per call. An EncoderGateway splits one call into
# ``max_batch_images``-sized requests run ``max_concurrency`` at a time, so this default keeps its
# 16 x 8 pipeline full; encoding one sample per call left it idle on a single round trip at a time.
ENCODE_GROUP_SIZE = 128


class _WritableStore(Protocol):
    def has(self, key: str) -> bool: ...
    def write(self, key: str, features: EncoderFeatures) -> None: ...
    def flush(self) -> None: ...


@dataclass
class PrecomputeStats:
    """Outcome of one :func:`precompute_features` call.

    Attributes
    ----------
    total : int
        Samples considered.
    computed : int
        Samples newly encoded and written.
    skipped : int
        Samples already present in the store — not re-encoded.
    failed : int
        Samples where encoding raised an :class:`~precisionai.agritune.encoder.errors.EncoderError`.
    failed_sample_ids : list[str]
        The ``sample_id`` of every failed sample, in encounter order.
    """

    total: int = 0
    computed: int = 0
    skipped: int = 0
    failed: int = 0
    failed_sample_ids: list[str] = field(default_factory=list)


async def precompute_features(
    samples: Sequence[PreparedSample],
    *,
    encoder: EncoderBackend,
    store: _WritableStore,
    encoder_fingerprint: EncoderFingerprint,
    image_hash_fn: Callable[[PreparedSample], str],
    on_progress: Callable[[PrecomputeStats], None] | None = None,
    group_size: int = ENCODE_GROUP_SIZE,
) -> PrecomputeStats:
    """Encode and store every sample in ``samples`` not already present in ``store``.

    Samples missing from the store are encoded ``group_size`` at a time in one encoder call and
    written one by one. If a group fails, its samples are retried individually, so one image the
    encoder rejects does not fail its neighbours.

    Parameters
    ----------
    samples : Sequence[PreparedSample]
        Samples to precompute features for (already augmented, if applicable).
    encoder : EncoderBackend
        Backend used to encode samples not already cached — typically an
        :class:`~precisionai.agritune.encoder.gateway.EncoderGateway`.
    store : DirectoryFeatureStore | ShardedFeatureStore
        Destination store; also consulted to decide what to skip.
    encoder_fingerprint : EncoderFingerprint
        Identifies the encoder configuration, for cache key derivation.
    image_hash_fn : Callable[[PreparedSample], str]
        Computes the image-content hash component of the cache key for one sample (e.g. hashing
        the sample's raw file bytes) — supplied by the caller since hashing strategy depends on
        the concrete image representation in use.
    on_progress : Callable[[PrecomputeStats], None] | None, optional
        Called after every sample with the running totals so far.
    group_size : int, optional
        Uncached samples per encoder call; pass an encoder that batches requests itself (an
        :class:`~precisionai.agritune.encoder.gateway.EncoderGateway`) when this is large.

    Returns
    -------
    PrecomputeStats
        Final totals. A non-empty ``failed_sample_ids`` means at least one sample could not be
        encoded — the caller decides whether that's acceptable for its use case.
    """
    if group_size < 1:
        raise ValueError(f"group_size must be positive; got {group_size}")
    stats = PrecomputeStats(total=len(samples))
    pending: list[tuple[PreparedSample, str]] = []

    for sample in samples:
        key = compute_feature_key(
            sample_id=sample.sample_id,
            image_hash=image_hash_fn(sample),
            augmentation_fingerprint=hash_augmentation(sample.augmentation_metadata),
            encoder_fingerprint=encoder_fingerprint,
        )
        if store.has(key):
            logger.debug("sample %s already cached, skipping", sample.sample_id)
            stats.skipped += 1
            if on_progress is not None:
                on_progress(stats)
            continue
        pending.append((sample, key))
        if len(pending) >= group_size:
            await _encode_group(pending, encoder=encoder, store=store, stats=stats, on_progress=on_progress)
            pending = []
    if pending:
        await _encode_group(pending, encoder=encoder, store=store, stats=stats, on_progress=on_progress)

    store.flush()
    logger.info(
        "precompute complete: total=%d computed=%d skipped=%d failed=%d",
        stats.total,
        stats.computed,
        stats.skipped,
        stats.failed,
    )
    return stats


async def _encode_group(
    group: list[tuple[PreparedSample, str]],
    *,
    encoder: EncoderBackend,
    store: _WritableStore,
    stats: PrecomputeStats,
    on_progress: Callable[[PrecomputeStats], None] | None,
) -> None:
    """Encode ``group`` in one encoder call and write each sample; fall back to one call per sample on failure."""
    try:
        features = await encoder.encode([sample.image for sample, _ in group])
    except EncoderError:
        if len(group) == 1:
            sample = group[0][0]
            logger.exception("failed to encode sample %s", sample.sample_id)
            stats.failed += 1
            stats.failed_sample_ids.append(sample.sample_id)
            if on_progress is not None:
                on_progress(stats)
            return
        logger.warning("encoding a group of %d sample(s) failed; retrying them one at a time", len(group))
        for item in group:
            await _encode_group([item], encoder=encoder, store=store, stats=stats, on_progress=on_progress)
        return
    for index, (_, key) in enumerate(group):
        store.write(key, select_one(features, index))
        stats.computed += 1
        if on_progress is not None:
            on_progress(stats)
