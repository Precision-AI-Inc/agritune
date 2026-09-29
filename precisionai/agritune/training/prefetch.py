# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Overlaps one batch's feature fetch with the previous batch's model compute.

``FeatureProvider.get_features`` — a disk/store read for ``cached``/``hybrid``, or an encoder call
for ``online`` — is synchronous. Without this, the training/evaluation loop blocks on it once per
batch no matter how many ``DataLoader`` workers are configured, since those workers only ever
produce ``TrainingBatch.samples``/``targets``, never features. ``PrefetchingFeatureLoader`` runs
one batch ahead in a single background thread, so while the GPU is busy in
``forward``/``backward`` for batch N, batch N+1's features are already being read and moved to
``device`` — keeping the GPU fed instead of idling between batches.

On a CUDA device the background thread issues its copies (and any device-side gather a provider
performs) on a dedicated CUDA stream. Work issued from another thread would otherwise still land
on the device's default stream and queue behind the training step it is meant to overlap. The
consumer waits on an event recorded after those copies before touching the tensors, and marks
them as used on its own stream so the caching allocator does not recycle their memory early.
"""

from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor

import torch

from precisionai.agritune.schemas.features import EncoderFeatures
from precisionai.agritune.schemas.protocols import FeatureProvider
from precisionai.agritune.training.batch import TrainingBatch

_Loaded = tuple[tuple[EncoderFeatures, torch.Tensor], "torch.cuda.Event | None"]


class PrefetchingFeatureLoader(Iterator[tuple[TrainingBatch, EncoderFeatures, torch.Tensor]]):
    """Yields ``(batch, features, targets)``, one batch's features fetched ahead of consumption.

    Parameters
    ----------
    batches : Iterable[TrainingBatch]
        Batches to fetch features for, in order. Iterated lazily, one ``next()`` ahead at a time —
        never buffers more than a single pending batch, so this does not change peak memory beyond
        one extra batch's features.
    feature_provider : FeatureProvider
        Supplies features for each batch's samples. Called from a single background thread, so
        calls never overlap each other — only the surrounding compute — meaning a provider that
        is not itself safe under concurrent ``get_features`` calls is still safe to use here.
    device : torch.device
        Where ``features``/``targets`` are moved before being handed back.
    non_blocking : bool
        Forwarded to every ``.to(device, non_blocking=...)`` call — should be ``True`` only when
        ``device.type == "cuda"`` and the caller's targets/features are pinned; a no-op otherwise.
    """

    def __init__(
        self,
        batches: Iterable[TrainingBatch],
        feature_provider: FeatureProvider,
        *,
        device: torch.device,
        non_blocking: bool,
    ) -> None:
        self._iterator = iter(batches)
        self._feature_provider = feature_provider
        self._device = device
        self._non_blocking = non_blocking
        self._copy_stream = torch.cuda.Stream(device) if device.type == "cuda" else None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending: tuple[TrainingBatch, Future[_Loaded]] | None = self._submit_next()

    def _submit_next(self) -> tuple[TrainingBatch, Future[_Loaded]] | None:
        try:
            batch = next(self._iterator)
        except StopIteration:
            return None
        return batch, self._executor.submit(self._load, batch)

    def _load(self, batch: TrainingBatch) -> _Loaded:
        if self._copy_stream is None:
            return self._fetch(batch), None
        with torch.cuda.stream(self._copy_stream):
            features, targets = self._fetch(batch)
            ready = torch.cuda.Event()
            ready.record(self._copy_stream)
        return (features, targets), ready

    def _fetch(self, batch: TrainingBatch) -> tuple[EncoderFeatures, torch.Tensor]:
        features = self._feature_provider.get_features(batch.samples)
        features = features.to(self._device, non_blocking=self._non_blocking)
        targets = batch.targets.to(self._device, non_blocking=self._non_blocking)
        return features, targets

    def _hand_over(self, features: EncoderFeatures, targets: torch.Tensor, ready: torch.cuda.Event | None) -> None:
        """Make the consuming stream wait for ``ready`` and own every tensor it is handed."""
        if ready is None:
            return
        consumer = torch.cuda.current_stream(self._device)
        consumer.wait_event(ready)
        for tensor in (
            features.patch_tokens,
            features.cls_tokens,
            features.patch_grid,
            features.valid_patch_mask,
            targets,
        ):
            if tensor is not None and tensor.device.type == "cuda":
                tensor.record_stream(consumer)

    def __iter__(self) -> "PrefetchingFeatureLoader":
        """Return ``self`` — this object is its own iterator."""
        return self

    def __next__(self) -> tuple[TrainingBatch, EncoderFeatures, torch.Tensor]:
        """Return the next ``(batch, features, targets)``, blocking only if the fetch isn't done yet."""
        if self._pending is None:
            self._executor.shutdown(wait=False)
            raise StopIteration
        batch, future = self._pending
        (features, targets), ready = future.result()
        self._hand_over(features, targets, ready)
        self._pending = self._submit_next()
        return batch, features, targets

    def __enter__(self) -> "PrefetchingFeatureLoader":
        """Return ``self`` — use as a context manager to guarantee the background thread is freed."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Shut down the background executor even if the consuming loop was abandoned early.

        ``__next__`` already shuts it down once the iterator is exhausted; calling
        ``shutdown`` again here is a no-op in that case. If the caller's loop instead exits via an
        exception or a ``break`` partway through, this is the only thing that still frees the
        worker thread — without it, the thread (and any in-flight fetch) would otherwise leak for
        the life of the process.
        """
        self._executor.shutdown(wait=False, cancel_futures=True)
