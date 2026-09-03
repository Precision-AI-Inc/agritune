# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Feature build orchestration — powers ``agritune features build``.

Build unaugmented features by default, or one deterministic offline augmentation variant when a
pipeline is supplied. Online/hybrid augmentation remains a training-time feature acquisition
mode and cannot be precomputed as a finite cache.
"""

from collections.abc import Callable
from io import BytesIO
from pathlib import Path

from PIL import Image

from precisionai.agritune.augmentations.image.pipeline import (
    AugmentationMode,
    ImageAugmentationPipeline,
    prepare_sample,
)
from precisionai.agritune.data.manifest import load_manifest
from precisionai.agritune.features.keys import EncoderFingerprint, hash_image_bytes
from precisionai.agritune.features.precompute import PrecomputeStats, precompute_features
from precisionai.agritune.features.store import DirectoryFeatureStore, ShardedFeatureStore
from precisionai.agritune.logging import get_logger
from precisionai.agritune.schemas.protocols import EncoderBackend
from precisionai.agritune.schemas.samples import PreparedSample, Sample

logger = get_logger(__name__)


async def build_features(
    manifest_path: str,
    *,
    store: DirectoryFeatureStore | ShardedFeatureStore,
    encoder: EncoderBackend,
    encoder_fingerprint: EncoderFingerprint,
    sample_ids: list[str] | None = None,
    augmentation_mode: AugmentationMode = AugmentationMode.NONE,
    augmentation_pipeline: ImageAugmentationPipeline | None = None,
    global_seed: int = 0,
    augmentation_variant: int = 0,
    on_progress: Callable[[PrecomputeStats], None] | None = None,
) -> PrecomputeStats:
    """Resumably precompute unaugmented or deterministic offline features from a manifest.

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
    augmentation_mode : AugmentationMode, optional
        ``NONE`` or ``OFFLINE``. Online/hybrid modes cannot be finitely precomputed.
    augmentation_pipeline : ImageAugmentationPipeline | None, optional
        Pipeline used under ``OFFLINE``; defaults to a no-op pipeline that still records the
        deterministic offline seed.
    global_seed : int, optional
        Seed used to derive each offline sample's augmentation seed.
    augmentation_variant : int, optional
        Fixed offline variant index to precompute.
    on_progress : Callable[[PrecomputeStats], None] | None, optional
        Called after every sample with running totals.

    Returns
    -------
    PrecomputeStats
    """
    if augmentation_mode not in (AugmentationMode.NONE, AugmentationMode.OFFLINE):
        raise ValueError(
            f"feature precomputation supports augmentation mode 'none' or 'offline'; got {augmentation_mode.value!r}"
        )

    manifest_dir = Path(manifest_path).parent
    rows = load_manifest(manifest_path)
    if sample_ids is not None:
        allowed = set(sample_ids)
        rows = [row for row in rows if row.sample_id in allowed]

    logger.info(
        "building features for %d sample(s): store=%s augmentation_mode=%s", len(rows), store, augmentation_mode.value
    )
    image_bytes_by_sample: dict[str, bytes] = {}
    samples: list[PreparedSample] = []
    for row in rows:
        data = (manifest_dir / row.image_path).read_bytes()
        image_bytes_by_sample[row.sample_id] = data
        image = Image.open(BytesIO(data)).convert("RGB")
        samples.append(PreparedSample(sample_id=row.sample_id, image=image, target=None))
        if augmentation_mode is AugmentationMode.NONE:
            continue

        with Image.open(manifest_dir / row.mask_path) as mask:
            source = Sample(sample_id=row.sample_id, image=image, target=mask.copy(), metadata=dict(row.metadata))
        prepared = prepare_sample(
            source,
            mode=AugmentationMode.OFFLINE,
            pipeline=augmentation_pipeline or ImageAugmentationPipeline(),
            global_seed=global_seed,
            variant=augmentation_variant,
        )
        samples.append(
            PreparedSample(
                sample_id=prepared.sample_id,
                image=prepared.image,
                target=None,
                augmentation_metadata=prepared.augmentation_metadata,
            )
        )

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
