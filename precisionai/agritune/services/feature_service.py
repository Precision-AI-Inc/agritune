# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Feature build orchestration — powers ``agritune features build``.

Build unaugmented features by default, or one deterministic offline augmentation variant when a
pipeline is supplied. Online/hybrid augmentation remains a training-time feature acquisition
mode and cannot be precomputed as a finite cache.

Manifest rows are read and decoded in bounded-size chunks rather than all at once — a manifest
with hundreds of thousands of rows would otherwise hold every decoded image in memory
simultaneously before the first encoder call.
"""

from collections.abc import Callable, Iterator
from io import BytesIO
from pathlib import Path

from PIL import Image

from precisionai.agritune.augmentations.image.pipeline import (
    AugmentationMode,
    ImageAugmentationPipeline,
    prepare_sample,
)
from precisionai.agritune.data.manifest import ManifestRow, load_manifest
from precisionai.agritune.features.keys import EncoderFingerprint, hash_image_bytes
from precisionai.agritune.features.precompute import PrecomputeStats, precompute_features
from precisionai.agritune.features.store import DirectoryFeatureStore, ShardedFeatureStore
from precisionai.agritune.logging import get_logger
from precisionai.agritune.schemas.protocols import EncoderBackend
from precisionai.agritune.schemas.samples import PreparedSample, Sample

logger = get_logger(__name__)

_BUILD_CHUNK_SIZE = 200  # bounds peak memory to ~this many decoded images, regardless of manifest size


def _iter_chunks(rows: list[ManifestRow], size: int) -> Iterator[list[ManifestRow]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _prepare_chunk(
    rows: list[ManifestRow],
    manifest_dir: Path,
    *,
    augmentation_mode: AugmentationMode,
    augmentation_pipeline: ImageAugmentationPipeline | None,
    global_seed: int,
    augmentation_variant: int,
) -> tuple[list[PreparedSample], dict[str, str]]:
    """Read and decode one chunk of manifest rows into prepared samples and their image hashes."""
    image_hash_by_sample: dict[str, str] = {}
    samples: list[PreparedSample] = []
    for row in rows:
        data = (manifest_dir / row.image_path).read_bytes()
        image_hash_by_sample[row.sample_id] = hash_image_bytes(data)
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
    return samples, image_hash_by_sample


def _make_image_hash_fn(image_hash_by_sample: dict[str, str]) -> Callable[[PreparedSample], str]:
    def image_hash_fn(sample: PreparedSample) -> str:
        return image_hash_by_sample[sample.sample_id]

    return image_hash_fn


def _make_progress_relay(
    total: int, aggregate: PrecomputeStats, on_progress: Callable[[PrecomputeStats], None] | None
) -> Callable[[PrecomputeStats], None] | None:
    """Translate one chunk's live progress into overall totals, for the caller's ``on_progress``."""
    if on_progress is None:
        return None

    def relay(chunk_stats: PrecomputeStats) -> None:
        on_progress(
            PrecomputeStats(
                total=total,
                computed=aggregate.computed + chunk_stats.computed,
                skipped=aggregate.skipped + chunk_stats.skipped,
                failed=aggregate.failed + chunk_stats.failed,
                failed_sample_ids=aggregate.failed_sample_ids + chunk_stats.failed_sample_ids,
            )
        )

    return relay


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
        Called after every sample with running totals across the whole manifest.

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

    total = len(rows) if augmentation_mode is AugmentationMode.NONE else len(rows) * 2
    logger.info(
        "building features for %d sample(s): store=%s augmentation_mode=%s", total, store, augmentation_mode.value
    )

    total_chunks = (len(rows) + _BUILD_CHUNK_SIZE - 1) // _BUILD_CHUNK_SIZE
    aggregate = PrecomputeStats(total=total)
    for chunk_index, chunk_rows in enumerate(_iter_chunks(rows, _BUILD_CHUNK_SIZE), start=1):
        logger.info(
            "reading and decoding chunk %d/%d (%d row(s)) before encoding",
            chunk_index,
            total_chunks,
            len(chunk_rows),
        )
        samples, image_hash_by_sample = _prepare_chunk(
            chunk_rows,
            manifest_dir,
            augmentation_mode=augmentation_mode,
            augmentation_pipeline=augmentation_pipeline,
            global_seed=global_seed,
            augmentation_variant=augmentation_variant,
        )
        chunk_result = await precompute_features(
            samples,
            encoder=encoder,
            store=store,
            encoder_fingerprint=encoder_fingerprint,
            image_hash_fn=_make_image_hash_fn(image_hash_by_sample),
            on_progress=_make_progress_relay(total, aggregate, on_progress),
        )
        aggregate.computed += chunk_result.computed
        aggregate.skipped += chunk_result.skipped
        aggregate.failed += chunk_result.failed
        aggregate.failed_sample_ids.extend(chunk_result.failed_sample_ids)

    return aggregate
