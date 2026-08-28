# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Feature build orchestration — powers ``agritune features build``.

No augmentation is applied here — this is the ``augmentation.mode: none`` / offline-cache path.
Augmented (offline-variant or online) feature building is future work; see
``agritune_implementation_plan.md`` §6/§26.
"""

from collections.abc import Callable
from io import BytesIO
from pathlib import Path

from PIL import Image

from precisionai.agritune.data.manifest import load_manifest
from precisionai.agritune.features.keys import EncoderFingerprint, hash_image_bytes
from precisionai.agritune.features.precompute import PrecomputeStats, precompute_features
from precisionai.agritune.features.store import DirectoryFeatureStore, ShardedFeatureStore
from precisionai.agritune.schemas.protocols import EncoderBackend
from precisionai.agritune.schemas.samples import PreparedSample


async def build_features(
    manifest_path: str,
    *,
    store: DirectoryFeatureStore | ShardedFeatureStore,
    encoder: EncoderBackend,
    encoder_fingerprint: EncoderFingerprint,
    sample_ids: list[str] | None = None,
    on_progress: Callable[[PrecomputeStats], None] | None = None,
) -> PrecomputeStats:
    """Resumably precompute (unaugmented) features for every sample in a manifest.

    Parameters
    ----------
    manifest_path : str
        Path to the dataset manifest CSV.
    store : FeatureStore
        Destination feature store; also consulted to skip already-computed samples.
    encoder : EncoderBackend
        Backend to encode samples not already cached (typically an
        :class:`~precisionai.agritune.encoder.gateway.EncoderGateway`).
    encoder_fingerprint : EncoderFingerprint
        Identifies the encoder configuration, for cache key derivation.
    sample_ids : list[str] | None, optional
        Restrict to these sample IDs (e.g. one split); ``None`` processes every manifest row.
    on_progress : Callable[[PrecomputeStats], None] | None, optional
        Called after every sample with running totals.

    Returns
    -------
    PrecomputeStats
    """
    manifest_dir = Path(manifest_path).parent
    rows = load_manifest(manifest_path)
    if sample_ids is not None:
        allowed = set(sample_ids)
        rows = [row for row in rows if row.sample_id in allowed]

    image_bytes_by_sample: dict[str, bytes] = {}
    samples: list[PreparedSample] = []
    for row in rows:
        data = (manifest_dir / row.image_path).read_bytes()
        image_bytes_by_sample[row.sample_id] = data
        image = Image.open(BytesIO(data)).convert("RGB")
        samples.append(PreparedSample(sample_id=row.sample_id, image=image, target=None))

    def image_hash_fn(sample: PreparedSample) -> str:
        return hash_image_bytes(image_bytes_by_sample[sample.sample_id])

    return await precompute_features(
        samples,
        encoder=encoder,
        store=store,
        encoder_fingerprint=encoder_fingerprint,
        image_hash_fn=image_hash_fn,
        on_progress=on_progress,
    )
