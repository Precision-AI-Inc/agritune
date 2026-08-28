# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""``CachedFeatureProvider`` — the ``FeatureProvider`` for offline-only augmentation modes.

Used when ``augmentation.mode`` is ``none`` or ``offline``, where every sample's features have
already been precomputed via ``agritune features build``. This is the only ``FeatureProvider``
implementation in v0.1 — see
``agritune_implementation_plan.md`` §26: ``OnlineFeatureProvider``/``HybridFeatureProvider`` are
v0.2 work (AGRITUNE-041/045), built only once offline training is stable. Nothing in
``precisionai.agritune.training`` or ``precisionai.agritune.tasks`` may import
``precisionai.agritune.encoder`` directly — every feature access goes through a
``FeatureProvider`` like this one.
"""

from collections.abc import Callable, Sequence

from precisionai.agritune.features.errors import FeatureNotCachedError
from precisionai.agritune.features.keys import EncoderFingerprint, compute_feature_key, hash_augmentation
from precisionai.agritune.schemas.features import EncoderFeatures, concatenate_encoder_features
from precisionai.agritune.schemas.protocols import FeatureStore
from precisionai.agritune.schemas.samples import PreparedSample


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

        per_sample_features = [self._read_one(sample) for sample in samples]
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
