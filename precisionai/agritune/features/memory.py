# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""``InMemoryFeatureProvider`` — serves cached features from one buffer loaded before training.

:class:`~precisionai.agritune.features.provider.CachedFeatureProvider` reads every batch from the
feature store: one ``store.read()`` per sample, a fresh :class:`EncoderFeatures` per sample, then a
concatenation, every epoch. For a linear-probe decoder that per-sample work, not the model, sets
the epoch time. This provider reads the store once, packs every requested sample into a single
contiguous ``(M, N, D)`` tensor (optionally cast to a smaller dtype such as ``float16``), validates
it once, and then serves each batch as a single ``index_select``.

The buffer can live in host memory (gathered into pinned memory per batch, then copied to the
compute device without blocking) or directly on a CUDA device (no per-batch copy at all). Like
every provider, this is the only thing the trainer sees; it never touches the encoder.
"""

import itertools
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch

from precisionai.agritune.features.errors import FeatureNotCachedError
from precisionai.agritune.features.store import DirectoryFeatureStore, ShardedFeatureStore
from precisionai.agritune.logging import get_logger, progress_iter
from precisionai.agritune.schemas.features import EncoderFeatures
from precisionai.agritune.schemas.samples import PreparedSample

logger = get_logger(__name__)

# Entries handed to one copy task while preloading, and copy tasks in flight at once.
_COPY_CHUNK = 64
_COPY_WORKERS = 8


@dataclass(frozen=True)
class _BufferLayout:
    num_rows: int
    max_patches: int
    patch_dim: int
    cls_dim: int | None
    encoder_model: str
    encoder_revision: str | None


def _chunks(items: Iterator[tuple[str, EncoderFeatures]], size: int) -> Iterator[list[tuple[str, EncoderFeatures]]]:
    chunk: list[tuple[str, EncoderFeatures]] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _finite_cast(tokens: torch.Tensor, dtype: torch.dtype, *, key: str, what: str) -> torch.Tensor:
    cast = tokens.to(dtype)
    if not torch.all(torch.isfinite(cast)):
        raise ValueError(
            f"entry {key!r} has non-finite {what} values after casting to {dtype} — use a wider "
            "feature_preload_dtype (e.g. bfloat16 or float32)"
        )
    return cast


class InMemoryFeatureProvider:
    """Serves features for a fixed set of samples from one preloaded, pre-validated buffer.

    Parameters
    ----------
    store : DirectoryFeatureStore | ShardedFeatureStore
        Already-populated store to preload from. Only read during construction.
    key_fn : Callable[[PreparedSample], str]
        Maps a sample to its store key — must be the same function a
        :class:`~precisionai.agritune.features.provider.CachedFeatureProvider` over ``store``
        would use (typically its ``key_for``).
    samples : Sequence[PreparedSample]
        Every sample :meth:`get_features` will later be asked for (e.g. the training and
        validation splits). Duplicates are loaded once.
    storage_device : torch.device
        Where the buffer lives: ``cpu`` (host RAM) or a CUDA device.
    output_device : torch.device
        Where :meth:`get_features` returns tensors. When ``storage_device`` is the CPU and this
        is a CUDA device, each batch is gathered into pinned memory and copied without blocking.
    storage_dtype : torch.dtype | None, optional
        Floating-point dtype the patch/CLS tokens are stored in; ``None`` keeps each store
        entry's own dtype. ``torch.float16`` halves memory and host-to-device traffic relative to
        ``float32``.
    output_dtype : torch.dtype | None, optional
        Floating-point dtype returned to the decoder, cast on ``output_device``; ``None`` returns
        the storage dtype unchanged. Defaults to ``torch.float32`` so a half-precision buffer
        still feeds a ``float32`` decoder.
    read_workers : int, optional
        Forwarded to the store's ``read_many`` (shards/files read concurrently while loading).
    show_progress : bool, optional
        Render a progress bar while loading.

    Raises
    ------
    ValueError
        If ``samples`` is empty, the entries disagree on encoder identity or embedding
        dimensions, or casting to ``storage_dtype`` produces non-finite values (e.g. a ``float16``
        overflow).
    KeyError
        If any sample has no entry in ``store``.
    """

    def __init__(
        self,
        store: DirectoryFeatureStore | ShardedFeatureStore,
        *,
        key_fn: Callable[[PreparedSample], str],
        samples: Sequence[PreparedSample],
        storage_device: torch.device,
        output_device: torch.device,
        storage_dtype: torch.dtype | None = None,
        output_dtype: torch.dtype | None = torch.float32,
        read_workers: int = 4,
        show_progress: bool = False,
    ) -> None:
        if not samples:
            raise ValueError("samples must be non-empty")
        self._key_fn = key_fn
        self._storage_device = storage_device
        self._output_device = output_device
        self._output_dtype = output_dtype
        self._pin_batches = storage_device.type == "cpu" and output_device.type == "cuda"
        # get_features is called with the same PreparedSample objects every epoch (PreloadedBatches
        # replays them), so their rows are memoized by identity; each entry keeps its sample
        # alive, so an id can never be reused by a different object while it is cached.
        self._row_by_identity: dict[int, tuple[PreparedSample, int]] = {}

        keys = list(dict.fromkeys(key_fn(sample) for sample in samples))
        self._row_of = {key: row for row, key in enumerate(keys)}
        layout = self._layout(store, keys)
        self._layout_info = layout
        self._load(
            store, keys, layout, storage_dtype=storage_dtype, read_workers=read_workers, show_progress=show_progress
        )

    @property
    def num_rows(self) -> int:
        """Return how many distinct entries are held in memory."""
        return self._layout_info.num_rows

    @property
    def nbytes(self) -> int:
        """Return the total size of the preloaded token buffers, in bytes."""
        total = self._patch_tokens.numel() * self._patch_tokens.element_size()
        if self._cls_tokens is not None:
            total += self._cls_tokens.numel() * self._cls_tokens.element_size()
        return total

    @staticmethod
    def _layout(store: DirectoryFeatureStore | ShardedFeatureStore, keys: list[str]) -> _BufferLayout:
        summaries = [store.read_summary(key) for key in keys]
        first = summaries[0]
        for summary in summaries[1:]:
            if summary.patch_dim != first.patch_dim or summary.cls_dim != first.cls_dim:
                raise ValueError(
                    f"entries disagree on embedding dimensions: {summary.key} has patch_dim="
                    f"{summary.patch_dim}, cls_dim={summary.cls_dim}; {first.key} has patch_dim="
                    f"{first.patch_dim}, cls_dim={first.cls_dim}"
                )
            if summary.encoder_model != first.encoder_model or summary.encoder_revision != first.encoder_revision:
                raise ValueError("entries disagree on encoder_model/encoder_revision")
        return _BufferLayout(
            num_rows=len(keys),
            max_patches=max(summary.num_patches for summary in summaries),
            patch_dim=first.patch_dim,
            cls_dim=first.cls_dim,
            encoder_model=first.encoder_model,
            encoder_revision=first.encoder_revision,
        )

    def _load(
        self,
        store: DirectoryFeatureStore | ShardedFeatureStore,
        keys: list[str],
        layout: _BufferLayout,
        *,
        storage_dtype: torch.dtype | None,
        read_workers: int,
        show_progress: bool,
    ) -> None:
        valid = torch.zeros(layout.num_rows, layout.max_patches, dtype=torch.bool)
        patch_grid = torch.zeros(layout.num_rows, 2, dtype=torch.long)
        self._image_sizes: list[tuple[int, int]] = [(0, 0)] * layout.num_rows

        entries = iter(
            progress_iter(
                store.read_many(keys, max_workers=read_workers),
                desc="preloading features",
                unit="sample",
                total=len(keys),
                disable=not show_progress,
            )
        )
        first = next(entries, None)
        if first is None:  # pragma: no cover — samples is non-empty and read_many yields every key
            raise RuntimeError("feature store yielded no entries")
        dtype = storage_dtype if storage_dtype is not None else first[1].patch_tokens.dtype
        patch_tokens = torch.zeros(layout.num_rows, layout.max_patches, layout.patch_dim, dtype=dtype)
        cls_tokens = torch.zeros(layout.num_rows, layout.cls_dim, dtype=dtype) if layout.cls_dim is not None else None

        def copy_chunk(chunk: list[tuple[str, EncoderFeatures]]) -> None:
            # Rows are disjoint across chunks, and these tensor ops release the GIL, so chunks
            # copy in parallel rather than serializing every cast and check on one thread.
            for key, features in chunk:
                row = self._row_of[key]
                tokens = features.patch_tokens[0]
                if features.valid_patch_mask is not None:
                    tokens = tokens[features.valid_patch_mask[0]]
                cast = _finite_cast(tokens, dtype, key=key, what="patch")
                patch_tokens[row, : cast.shape[0]] = cast
                valid[row, : cast.shape[0]] = True
                if cls_tokens is not None and features.cls_tokens is not None:
                    cls_tokens[row] = _finite_cast(features.cls_tokens[0], dtype, key=key, what="CLS")
                patch_grid[row] = features.patch_grid[0]
                self._image_sizes[row] = features.image_sizes[0]

        with ThreadPoolExecutor(max_workers=_COPY_WORKERS) as executor:
            in_flight: deque[Future[None]] = deque()
            for chunk in _chunks(itertools.chain([first], entries), _COPY_CHUNK):
                in_flight.append(executor.submit(copy_chunk, chunk))
                if len(in_flight) >= _COPY_WORKERS:
                    in_flight.popleft().result()
            while in_flight:
                in_flight.popleft().result()

        # Host-side copies of the per-row patch counts decide padding and masks per batch without
        # reading anything back from the device.
        self._num_valid = valid.sum(dim=1).numpy()
        self._any_padding = bool((self._num_valid < layout.max_patches).any())
        self._patch_tokens = patch_tokens.to(self._storage_device)
        self._cls_tokens = cls_tokens.to(self._storage_device) if cls_tokens is not None else None
        self._valid = valid.to(self._storage_device) if self._any_padding else None
        self._patch_grid = patch_grid.to(self._storage_device)
        logger.info(
            "preloaded %d feature entries (%d x %d x %d, %s) into %s: %.2f GB",
            layout.num_rows,
            layout.num_rows,
            layout.max_patches,
            layout.patch_dim,
            self._patch_tokens.dtype,
            self._storage_device,
            self.nbytes / 1e9,
        )

    def get_features(self, samples: Sequence[PreparedSample]) -> EncoderFeatures:
        """Return preloaded features for every sample in ``samples``, in order.

        Matches :func:`~precisionai.agritune.schemas.features.concatenate_encoder_features`'s
        batching: tokens are padded to the largest patch count in *this* batch, and
        ``valid_patch_mask`` is ``None`` unless the batch actually mixes patch counts.

        Parameters
        ----------
        samples : Sequence[PreparedSample]
            Samples to look up. Must be non-empty and among the samples preloaded at construction.

        Returns
        -------
        EncoderFeatures
            On ``output_device``, tokens in ``output_dtype``.

        Raises
        ------
        ValueError
            If ``samples`` is empty.
        FeatureNotCachedError
            If a sample was not among the preloaded samples.
        """
        if not samples:
            raise ValueError("samples must be non-empty")
        rows = self._rows(samples)
        counts = self._num_valid[rows]
        batch_patches = int(counts.max())
        needs_mask = bool(counts.min() != batch_patches)

        index = torch.from_numpy(rows)
        if self._storage_device.type != "cpu":
            index = index.to(self._storage_device, non_blocking=True)
        patch_tokens = self._gather(self._patch_tokens[:, :batch_patches], index)
        cls_tokens = self._gather(self._cls_tokens, index) if self._cls_tokens is not None else None
        valid = self._gather(self._valid[:, :batch_patches], index) if needs_mask and self._valid is not None else None
        patch_grid = self._gather(self._patch_grid, index)

        return EncoderFeatures.trusted(
            patch_tokens=self._to_output(patch_tokens, cast=True),
            cls_tokens=self._to_output(cls_tokens, cast=True) if cls_tokens is not None else None,
            patch_grid=self._to_output(patch_grid, cast=False),
            valid_patch_mask=self._to_output(valid, cast=False) if valid is not None else None,
            image_sizes=[self._image_sizes[row] for row in rows.tolist()],
            encoder_model=self._layout_info.encoder_model,
            encoder_revision=self._layout_info.encoder_revision,
        )

    def _rows(self, samples: Sequence[PreparedSample]) -> np.ndarray:
        rows = np.empty(len(samples), dtype=np.int64)
        for position, sample in enumerate(samples):
            cached = self._row_by_identity.get(id(sample))
            if cached is not None and cached[0] is sample:
                rows[position] = cached[1]
                continue
            key = self._key_fn(sample)
            row = self._row_of.get(key)
            if row is None:
                raise FeatureNotCachedError(
                    f"sample '{sample.sample_id}' (key={key}) was not preloaded — InMemoryFeatureProvider only "
                    "serves the samples it was constructed with"
                )
            self._row_by_identity[id(sample)] = (sample, row)
            rows[position] = row
        return rows

    def _gather(self, source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        if not self._pin_batches:
            return source.index_select(0, index)
        # Pinned host memory from PyTorch's caching host allocator: reused across batches, and
        # safe to reuse only once the asynchronous copy reading from it has finished.
        out = torch.empty((index.shape[0], *source.shape[1:]), dtype=source.dtype, pin_memory=True)
        return torch.index_select(source, 0, index, out=out)

    def _to_output(self, tensor: torch.Tensor, *, cast: bool) -> torch.Tensor:
        moved = tensor.to(self._output_device, non_blocking=self._pin_batches)
        if cast and self._output_dtype is not None:
            moved = moved.to(self._output_dtype)
        return moved
