# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""``FeatureProvider`` implementations: ``CachedFeatureProvider``, ``OnlineFeatureProvider``, ``HybridFeatureProvider``.

``CachedFeatureProvider`` is used when ``augmentation.mode`` is ``none`` or ``offline``, where
every sample's features have already been precomputed via ``agritune features build``.
``OnlineFeatureProvider`` encodes every batch fresh through an
:class:`~precisionai.agritune.encoder.gateway.EncoderGateway`, for ``augmentation.mode: online``.
``HybridFeatureProvider`` reads from a store when possible and falls back to the gateway
(write-through) on a miss. Nothing in ``precisionai.agritune.training`` or
``precisionai.agritune.tasks`` may import ``precisionai.agritune.encoder`` directly — every
feature access goes through a ``FeatureProvider`` like these.
"""

import asyncio
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

from precisionai.agritune.encoder.gateway import EncoderGateway
from precisionai.agritune.features.errors import FeatureNotCachedError
from precisionai.agritune.features.keys import EncoderFingerprint, compute_feature_key, hash_augmentation
from precisionai.agritune.features.store import DirectoryFeatureStore, ShardedFeatureStore
from precisionai.agritune.schemas.features import EncoderFeatures, concatenate_encoder_features, select_one
from precisionai.agritune.schemas.protocols import FeatureStore
from precisionai.agritune.schemas.samples import PreparedSample

_MAX_READ_WORKERS = 32


class CachedFeatureProvider:
    """Reads precomputed features out of a :class:`~precisionai.agritune.schemas.protocols.FeatureStore`.

    Parameters
    ----------
    store : FeatureStore
        The store to read from (typically a
        :class:`~precisionai.agritune.features.store.DirectoryFeatureStore` or
        :class:`~precisionai.agritune.features.store.ShardedFeatureStore`, already populated by
        ``agritune features build``).
    encoder_fingerprint : EncoderFingerprint
        Identifies the encoder configuration, for cache key derivation — must match what was used
        to build the store.
    image_hash_fn : Callable[[PreparedSample], str]
        Computes the image-content hash component of the cache key for one sample; must match
        what was used during precomputation.
    """

    def __init__(
        self,
        store: FeatureStore,
        *,
        encoder_fingerprint: EncoderFingerprint,
        image_hash_fn: Callable[[PreparedSample], str],
    ) -> None:
        self._store = store
        self._encoder_fingerprint = encoder_fingerprint
        self._image_hash_fn = image_hash_fn

    def get_features(self, samples: Sequence[PreparedSample]) -> EncoderFeatures:
        """Return cached features for every sample in ``samples``, in order.

        Reads are issued from a thread pool rather than sequentially: each is a file open plus a
        safetensors deserialization, which releases the GIL for the actual I/O, so concurrent reads
        overlap disk/filesystem latency instead of paying it once per sample in series — the
        dominant cost at the batch sizes this is called with.

        Parameters
        ----------
        samples : Sequence[PreparedSample]
            Samples to look up. Must be non-empty.

        Returns
        -------
        EncoderFeatures
            Batched features, one entry per sample.

        Raises
        ------
        ValueError
            If ``samples`` is empty.
        FeatureNotCachedError
            If any sample has no matching entry in the store — training must not silently fall
            back to an encoder call here; that is what an online/hybrid provider is for.
        """
        if not samples:
            raise ValueError("samples must be non-empty")

        workers = min(_MAX_READ_WORKERS, len(samples))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            per_sample_features = list(executor.map(self._read_one, samples))
        return concatenate_encoder_features(per_sample_features)

    def _read_one(self, sample: PreparedSample) -> EncoderFeatures:
        key = self.key_for(sample)
        try:
            return self._store.read(key)
        except KeyError as exc:
            raise FeatureNotCachedError(
                f"no cached features for sample '{sample.sample_id}' (key={key}); run "
                "`agritune features build` first, or configure an online/hybrid feature provider"
            ) from exc

    def key_for(self, sample: PreparedSample) -> str:
        """Compute the cache key for ``sample`` under this provider's fingerprint.

        Exposed so ``agritune features build`` can key its writes identically to how this
        provider will look them up.
        """
        return compute_feature_key(
            sample_id=sample.sample_id,
            image_hash=self._image_hash_fn(sample),
            augmentation_fingerprint=hash_augmentation(sample.augmentation_metadata),
            encoder_fingerprint=self._encoder_fingerprint,
        )


class OnlineFeatureProvider:
    """Encodes every batch fresh through an :class:`EncoderGateway` — no disk cache at all.

    Used when ``augmentation.mode: online`` and ``features.provider: online``: samples are
    augmented differently every epoch, so a disk cache would never hit anyway. Bridges the
    gateway's ``async`` ``encode()`` to the synchronous ``FeatureProvider`` interface the trainer
    expects via ``asyncio.run``, the same pattern used to drive gateway calls from the CLI's
    synchronous command handlers (``agritune features build``, ``agritune encoder benchmark``).

    Parameters
    ----------
    gateway : EncoderGateway
        Gateway to encode through — provides batching, rate limiting, retries, and validation.
    """

    def __init__(self, gateway: EncoderGateway) -> None:
        self._gateway = gateway

    def get_features(self, samples: Sequence[PreparedSample]) -> EncoderFeatures:
        """Encode every sample in ``samples`` through the gateway, in order.

        Parameters
        ----------
        samples : Sequence[PreparedSample]
            Samples to encode. Must be non-empty.

        Returns
        -------
        EncoderFeatures
            Batched features for every sample.

        Raises
        ------
        ValueError
            If ``samples`` is empty.
        """
        if not samples:
            raise ValueError("samples must be non-empty")
        return asyncio.run(self._gateway.encode([sample.image for sample in samples]))


class HybridFeatureProvider:
    """Reads from a store when possible, falling back to an encoder gateway on a cache miss.

    A miss is encoded and written back (write-through), so a hybrid run against a
    partially-precomputed :class:`~precisionai.agritune.schemas.protocols.FeatureStore` only ever
    encodes what it has not already seen.

    Parameters
    ----------
    store : DirectoryFeatureStore | ShardedFeatureStore
        Store consulted first, and written to on every miss.
    gateway : EncoderGateway
        Gateway used to encode samples missing from ``store``.
    encoder_fingerprint : EncoderFingerprint
        Identifies the encoder configuration, for cache key derivation — must match what
        ``gateway`` actually encodes with.
    image_hash_fn : Callable[[PreparedSample], str]
        Computes the image-content hash component of the cache key for one sample; must match
        what any offline precomputation of the same store used.
    """

    def __init__(
        self,
        store: DirectoryFeatureStore | ShardedFeatureStore,
        *,
        gateway: EncoderGateway,
        encoder_fingerprint: EncoderFingerprint,
        image_hash_fn: Callable[[PreparedSample], str],
    ) -> None:
        self._store = store
        self._gateway = gateway
        self._cached = CachedFeatureProvider(
            store, encoder_fingerprint=encoder_fingerprint, image_hash_fn=image_hash_fn
        )

    def get_features(self, samples: Sequence[PreparedSample]) -> EncoderFeatures:
        """Return features for every sample in ``samples``, in order.

        Cache hits are read from the store; misses are encoded together in one gateway call, each
        written back to the store under its own key, then merged back into the returned order.

        Parameters
        ----------
        samples : Sequence[PreparedSample]
            Samples to obtain features for. Must be non-empty.

        Returns
        -------
        EncoderFeatures
            Batched features, one entry per sample, in the same order as ``samples``.

        Raises
        ------
        ValueError
            If ``samples`` is empty.
        """
        if not samples:
            raise ValueError("samples must be non-empty")

        resolved: dict[int, EncoderFeatures] = {}
        misses: list[tuple[int, PreparedSample]] = []
        for index, sample in enumerate(samples):
            hit = self._try_read_cached(sample)
            if hit is None:
                misses.append((index, sample))
            else:
                resolved[index] = hit

        if misses:
            miss_samples = [sample for _, sample in misses]
            encoded = asyncio.run(self._gateway.encode([sample.image for sample in miss_samples]))
            for position, (index, sample) in enumerate(misses):
                one = select_one(encoded, position)
                self._store.write(self._cached.key_for(sample), one)
                resolved[index] = one

        return concatenate_encoder_features([resolved[index] for index in range(len(samples))])

    def key_for(self, sample: PreparedSample) -> str:
        """Compute the cache key ``sample`` is read from / written under in the wrapped store."""
        return self._cached.key_for(sample)

    def _try_read_cached(self, sample: PreparedSample) -> EncoderFeatures | None:
        try:
            return self._cached.get_features([sample])
        except FeatureNotCachedError:
            return None
