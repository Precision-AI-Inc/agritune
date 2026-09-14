# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the segmentation services.

Used by :mod:`training_service`, :mod:`evaluation_service`, and :mod:`prediction_service`:
building batches, decoders, and a
:class:`~precisionai.agritune.features.provider.CachedFeatureProvider` keyed consistently with
what ``agritune features build`` wrote.
"""

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from precisionai.agritune.augmentations.image.pipeline import (
    AugmentationMode,
    AugmentationPipelineConfig,
    GeometricConfig,
    ImageAugmentationPipeline,
    prepare_sample,
)
from precisionai.agritune.data.dataset import ManifestDataset
from precisionai.agritune.data.manifest import ManifestRow, load_manifest
from precisionai.agritune.features.keys import EncoderFingerprint, hash_image_bytes
from precisionai.agritune.features.provider import CachedFeatureProvider
from precisionai.agritune.logging import get_logger, progress_iter
from precisionai.agritune.schemas.protocols import FeatureProvider, FeatureStore
from precisionai.agritune.schemas.samples import PreparedSample
from precisionai.agritune.tasks.segmentation.decoders.aspp import ASPPDecoder
from precisionai.agritune.tasks.segmentation.decoders.mask_former import MaskFormerDecoder
from precisionai.agritune.tasks.segmentation.decoders.mlp_probe import MLPProbeDecoder
from precisionai.agritune.tasks.segmentation.decoders.pyramid_pooling import PyramidPoolingDecoder
from precisionai.agritune.tasks.segmentation.decoders.segmenter import SegmenterMaskTransformerDecoder
from precisionai.agritune.tasks.segmentation.decoders.token_fpn import CLSFusion, TokenFPNDecoder
from precisionai.agritune.training.evaluator import TrainingBatch

_DECODER_BUILDERS: dict[str, Callable[..., nn.Module]] = {
    "mlp_probe": MLPProbeDecoder,
    "aspp": ASPPDecoder,
    "ppm": PyramidPoolingDecoder,
    "segmenter": SegmenterMaskTransformerDecoder,
    "mask_former": MaskFormerDecoder,
}
logger = get_logger(__name__)


def mask_to_target_tensor(mask: Any) -> torch.Tensor:
    """Convert a decoded mask image to an integer class-index tensor, shape ``(H, W)``."""
    return torch.from_numpy(np.array(mask)).long()


def mask_output_size(mask: Any) -> tuple[int, int]:
    """Return a decoded mask's ``(height, width)`` as a concrete 2-tuple for :func:`build_decoder`.

    ``np.ndarray.shape`` is typed as a variable-length ``tuple[int, ...]``, which does not satisfy
    ``build_decoder``'s ``output_size: tuple[int, int]`` under static type checking even though a
    2D mask always produces exactly two dimensions at runtime.
    """
    height, width = np.array(mask).shape
    return int(height), int(width)


class _ResizedBatches:
    """``TrainingBatch`` iterable produced by :func:`build_training_batches`; see its docstring."""

    def __init__(
        self,
        dataset: ManifestDataset,
        *,
        batch_size: int,
        resize: tuple[int, int] | None,
        show_progress: bool,
    ) -> None:
        self._dataset = dataset
        self._batch_size = batch_size
        self._resize_pipeline = (
            ImageAugmentationPipeline(AugmentationPipelineConfig(geometric=GeometricConfig(resize=resize)))
            if resize is not None
            else None
        )
        self._show_progress = show_progress

    def __iter__(self) -> Iterator[TrainingBatch]:
        starts = progress_iter(
            range(0, len(self._dataset), self._batch_size),
            desc="building batches",
            unit="batch",
            disable=not self._show_progress,
        )
        for start in starts:
            chunk = [self._dataset[i] for i in range(start, min(start + self._batch_size, len(self._dataset)))]
            if self._resize_pipeline is not None:
                chunk = [self._resize_pipeline.apply(sample, seed=0) for sample in chunk]
            samples = [PreparedSample(sample_id=sample.sample_id, image=sample.image, target=None) for sample in chunk]
            targets = torch.stack([mask_to_target_tensor(sample.target) for sample in chunk])
            yield TrainingBatch(samples=samples, targets=targets)


def build_training_batches(
    dataset: ManifestDataset,
    *,
    batch_size: int,
    resize: tuple[int, int] | None = None,
    show_progress: bool = False,
) -> Iterable[TrainingBatch]:
    """Return a :class:`TrainingBatch` iterable, chunked to at most ``batch_size`` samples.

    Every image/mask is read from disk (and resized, if requested) lazily, one chunk at a time, on
    every iteration — nothing is decoded up front, and nothing from a previous chunk is held once
    the next one is yielded, so a dataset of any size can be iterated without loading it into
    memory all at once. The returned object can be iterated more than once (each ``__iter__`` call
    starts a fresh pass reading from disk): :class:`~precisionai.agritune.training.trainer.Trainer`
    relies on that to re-walk the same training/validation batches every epoch. Pass
    ``show_progress=True`` to render a bar over each pass rather than leaving the caller looking
    frozen while it reads.

    Parameters
    ----------
    dataset : ManifestDataset
    batch_size : int
    resize : tuple[int, int] | None, optional
        ``(width, height)`` every sample's image and mask are deterministically resized to before
        stacking — dimensional normalization, not augmentation, so it applies unconditionally (no
        flips/crops/photometric transforms are involved). Required whenever the dataset's own
        images/masks do not already share one native size, since :func:`torch.stack` cannot batch
        unequal-size targets together. ``None`` (the default) stacks each sample's native-size mask
        as-is, which only succeeds if every sample in the dataset already shares one size.
    show_progress : bool, optional
        Render a ``tqdm`` bar over each pass. Defaults to ``False`` so headless callers (e.g. the
        API) see no terminal output.
    """
    return _ResizedBatches(dataset, batch_size=batch_size, resize=resize, show_progress=show_progress)


class _StaticAugmentedBatches:
    """``TrainingBatch`` iterable produced by :func:`build_static_augmented_batches`; see its docstring."""

    def __init__(
        self,
        dataset: ManifestDataset,
        *,
        pipeline: ImageAugmentationPipeline,
        mode: AugmentationMode,
        global_seed: int,
        batch_size: int,
        variant: int,
        show_progress: bool,
    ) -> None:
        self._dataset = dataset
        self._pipeline = pipeline
        self._mode = mode
        self._global_seed = global_seed
        self._batch_size = batch_size
        self._variant = variant
        self._show_progress = show_progress

    def __iter__(self) -> Iterator[TrainingBatch]:
        starts = progress_iter(
            range(0, len(self._dataset), self._batch_size),
            desc="building batches",
            unit="batch",
            disable=not self._show_progress,
        )
        for start in starts:
            chunk = [self._dataset[i] for i in range(start, min(start + self._batch_size, len(self._dataset)))]
            prepared = [
                prepare_sample(
                    sample,
                    mode=self._mode,
                    pipeline=self._pipeline,
                    global_seed=self._global_seed,
                    variant=self._variant,
                )
                for sample in chunk
            ]
            samples = [
                PreparedSample(
                    sample_id=p.sample_id,
                    image=p.image,
                    target=None,
                    augmentation_metadata=p.augmentation_metadata,
                )
                for p in prepared
            ]
            targets = torch.stack([mask_to_target_tensor(p.target) for p in prepared])
            yield TrainingBatch(samples=samples, targets=targets)


def build_static_augmented_batches(
    dataset: ManifestDataset,
    *,
    pipeline: ImageAugmentationPipeline,
    mode: AugmentationMode,
    global_seed: int,
    batch_size: int,
    variant: int = 0,
    show_progress: bool = False,
) -> Iterable[TrainingBatch]:
    """Return a batches iterable applying a fixed (non-epoch-varying) augmentation pass.

    Suitable for ``AugmentationMode.NONE`` (a no-op passthrough) and ``AugmentationMode.OFFLINE``
    (one fixed ``variant`` per sample) — the same deterministic augmentation is recomputed every
    epoch, same as :func:`build_training_batches`. Use :class:`OnlineAugmentedBatches` instead for
    ``ONLINE``/``HYBRID``, which must re-augment differently every epoch.

    Every image/mask is read from disk and augmented lazily, one chunk at a time, on every
    iteration — nothing is decoded up front, and nothing from a previous chunk is held once the
    next one is yielded, so a dataset of any size can be iterated without loading it into memory
    all at once. The returned object can be iterated more than once (each ``__iter__`` call starts
    a fresh pass reading from disk, recomputing the same deterministic augmentation):
    :class:`~precisionai.agritune.training.trainer.Trainer` relies on that to re-walk the same
    training batches every epoch. Pass ``show_progress=True`` to render a bar over each pass rather
    than leaving the caller looking frozen while it reads.

    Parameters
    ----------
    dataset : ManifestDataset
    pipeline : ImageAugmentationPipeline
        Ignored when ``mode is AugmentationMode.NONE``.
    mode : AugmentationMode
        Must be ``NONE`` or ``OFFLINE``.
    global_seed : int
    batch_size : int
    variant : int, optional
        Offline variant index; ignored under ``NONE``.
    show_progress : bool, optional
        Render a ``tqdm`` bar over each pass. Defaults to ``False`` so headless callers (e.g. the
        API) see no terminal output.

    Returns
    -------
    Iterable[TrainingBatch]
    """
    return _StaticAugmentedBatches(
        dataset,
        pipeline=pipeline,
        mode=mode,
        global_seed=global_seed,
        batch_size=batch_size,
        variant=variant,
        show_progress=show_progress,
    )


class OnlineAugmentedBatches:
    """A batches iterable that re-augments its samples fresh every time it is iterated.

    ``Trainer.fit()`` iterates its ``train_batches`` argument once per epoch via a plain ``for
    batch in batches`` loop — passing an instance of this class as that argument makes each of
    those per-epoch iterations produce newly re-augmented ``ONLINE``/``HYBRID`` batches, with no
    change to ``Trainer`` itself. Not re-iterable concurrently: each ``__iter__`` call consumes and
    advances the shared epoch counter.

    Parameters
    ----------
    dataset : ManifestDataset
    pipeline : ImageAugmentationPipeline
    mode : AugmentationMode
        Must be ``ONLINE`` or ``HYBRID``.
    global_seed : int
    batch_size : int
    hybrid_online_probability : float, optional
        Forwarded to :func:`~precisionai.agritune.augmentations.image.pipeline.prepare_sample`;
        ignored under ``ONLINE``.
    """

    def __init__(
        self,
        dataset: ManifestDataset,
        *,
        pipeline: ImageAugmentationPipeline,
        mode: AugmentationMode,
        global_seed: int,
        batch_size: int,
        hybrid_online_probability: float = 0.3,
    ) -> None:
        self._dataset = dataset
        self._pipeline = pipeline
        self._mode = mode
        self._global_seed = global_seed
        self._batch_size = batch_size
        self._hybrid_online_probability = hybrid_online_probability
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the next epoch index, for exact checkpoint resume."""
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative; got {epoch}")
        self._epoch = epoch

    def __iter__(self) -> Iterator[TrainingBatch]:
        """Yield one freshly re-augmented pass over the dataset, advancing the epoch counter."""
        epoch = self._epoch
        self._epoch += 1
        for start in range(0, len(self._dataset), self._batch_size):
            chunk = [self._dataset[i] for i in range(start, min(start + self._batch_size, len(self._dataset)))]
            prepared = [
                prepare_sample(
                    sample,
                    mode=self._mode,
                    pipeline=self._pipeline,
                    global_seed=self._global_seed,
                    epoch=epoch,
                    occurrence=0,
                    hybrid_online_probability=self._hybrid_online_probability,
                )
                for sample in chunk
            ]
            samples = [
                PreparedSample(
                    sample_id=p.sample_id,
                    image=p.image,
                    target=None,
                    augmentation_metadata=p.augmentation_metadata,
                )
                for p in prepared
            ]
            targets = torch.stack([mask_to_target_tensor(p.target) for p in prepared])
            yield TrainingBatch(samples=samples, targets=targets)


def build_prediction_batches(dataset: ManifestDataset, *, batch_size: int) -> Iterator[list[PreparedSample]]:
    """Yield sample batches with no target tensor required (for inference), one chunk at a time.

    Unlike :func:`build_training_batches`/:func:`build_static_augmented_batches` — which return a
    materialized list because training reuses the same batches every epoch — prediction consumes
    each batch exactly once and writes its output immediately, so this yields lazily instead of
    decoding every sample's image into memory up front. That matters at prediction scale: a
    dataset of hundreds of thousands of samples would otherwise hold every decoded image
    simultaneously before the first prediction is even written.
    """
    for start in range(0, len(dataset), batch_size):
        chunk = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        yield [PreparedSample(sample_id=sample.sample_id, image=sample.image, target=None) for sample in chunk]


def build_decoder(
    name: str,
    *,
    patch_dim: int,
    cls_dim: int | None,
    num_classes: int,
    output_size: tuple[int, int],
    **kwargs: Any,
) -> nn.Module:
    """Construct the named decoder.

    Parameters
    ----------
    name : str
        ``"mlp_probe"``, ``"token_fpn"``, ``"aspp"``, ``"ppm"``, ``"segmenter"``, or
        ``"mask_former"``.
    patch_dim : int
        Patch embedding dimension.
    cls_dim : int | None
        CLS embedding dimension; only consulted by ``"token_fpn"``.
    num_classes : int
        Number of segmentation classes.
    output_size : tuple[int, int]
        ``(height, width)`` to upsample logits to.
    **kwargs : Any
        Forwarded to the selected decoder's constructor — e.g. ``hidden_dims`` for
        ``"mlp_probe"``, ``atrous_rates`` for ``"aspp"``, ``pool_sizes`` for ``"ppm"``,
        ``num_queries`` for ``"mask_former"``, or ``cls_fusion``/``hidden_dim`` for ``"token_fpn"``.

    Returns
    -------
    torch.nn.Module

    Raises
    ------
    ValueError
        If ``name`` is not one of the supported decoders.
    """
    logger.debug(
        "building decoder %r: patch_dim=%d cls_dim=%s num_classes=%d output_size=%s",
        name,
        patch_dim,
        cls_dim,
        num_classes,
        output_size,
    )
    if name == "token_fpn":
        if "cls_fusion" in kwargs:
            kwargs = {**kwargs, "cls_fusion": CLSFusion(kwargs["cls_fusion"])}
        return TokenFPNDecoder(
            patch_dim=patch_dim, num_classes=num_classes, output_size=output_size, cls_dim=cls_dim, **kwargs
        )

    decoder_cls = _DECODER_BUILDERS.get(name)
    if decoder_cls is None:
        raise ValueError(f"unsupported decoder: {name!r}")
    return decoder_cls(patch_dim=patch_dim, num_classes=num_classes, output_size=output_size, **kwargs)


def probe_feature_dims(provider: FeatureProvider, sample: PreparedSample) -> tuple[int, int | None]:
    """Return ``(patch_dim, cls_dim)`` observed from one sample's cached features."""
    features = provider.get_features([sample])
    patch_dim = features.patch_tokens.shape[-1]
    cls_dim = features.cls_tokens.shape[-1] if features.cls_tokens is not None else None
    return patch_dim, cls_dim


def build_image_hash_fn(manifest_path: str) -> Callable[[PreparedSample], str]:
    """Return a function computing a sample's image-content hash, keyed by manifest row.

    The feature store is built by ``feature_service.build_features`` using file-content hashes;
    any provider reading or writing that same store must derive keys identically — shared here so
    :class:`~precisionai.agritune.features.provider.CachedFeatureProvider` and
    :class:`~precisionai.agritune.features.provider.HybridFeatureProvider` never duplicate it.

    Parameters
    ----------
    manifest_path : str
        Path to the dataset manifest CSV.

    Returns
    -------
    Callable[[PreparedSample], str]
    """
    rows = load_manifest(manifest_path)
    base_dir = Path(manifest_path).parent
    rows_by_sample_id = {row.sample_id: row for row in rows}

    def image_hash_fn(sample: PreparedSample) -> str:
        row = rows_by_sample_id[sample.sample_id]
        return hash_image_bytes((base_dir / row.image_path).read_bytes())

    return image_hash_fn


def build_cached_feature_provider(
    manifest_path: str, *, store: FeatureStore, encoder_fingerprint: EncoderFingerprint
) -> tuple[CachedFeatureProvider, list[ManifestRow]]:
    """Build a :class:`CachedFeatureProvider` keyed identically to ``agritune features build``.

    Returns
    -------
    tuple[CachedFeatureProvider, list[ManifestRow]]
        The provider, plus the parsed manifest rows (callers typically need these for splitting).
    """
    rows = load_manifest(manifest_path)
    provider = CachedFeatureProvider(
        store, encoder_fingerprint=encoder_fingerprint, image_hash_fn=build_image_hash_fn(manifest_path)
    )
    return provider, rows
