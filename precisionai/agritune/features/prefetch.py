# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""``PrefetchingFeatureProvider`` — overlaps encoding with training via a bounded background queue.

Desired execution::

    CPU            prepare batch N+2
    Network        encode batch N+1
    GPU            train batch N

A dedicated background thread runs ahead of the training loop, calling an inner
:class:`~precisionai.agritune.features.provider.OnlineFeatureProvider` (or
:class:`~precisionai.agritune.features.provider.HybridFeatureProvider`) for upcoming batches and
buffering the results in a bounded queue, so ``Trainer``'s synchronous ``get_features`` call
usually finds a batch's features already sitting in the queue rather than blocking on the
network. Never calls the encoder from a regular ``DataLoader`` worker — this is that dedicated
feature-acquisition thread.
"""

import queue
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from precisionai.agritune.schemas.features import EncoderFeatures
from precisionai.agritune.schemas.protocols import FeatureProvider
from precisionai.agritune.schemas.samples import PreparedSample


@dataclass
class _Success:
    sample_ids: list[str]
    features: EncoderFeatures


@dataclass
class _Failure:
    error: Exception


class _Done:
    pass


_QueueItem = _Success | _Failure | _Done


class PrefetchingFeatureProvider:
    """Stays ahead of the training loop by encoding upcoming batches on a background thread.

    Must be given the *entire* upcoming sequence of batches at construction time, in the exact
    order :meth:`get_features` will be called for them (e.g. once through a set of offline
    batches, or several full passes chained together for online training across epochs) — the
    background thread has no other way to know what is coming next. Each call to
    :meth:`get_features` is checked against that expected order and raises if it does not match,
    to catch integration mistakes early rather than silently returning the wrong batch's features.

    Parameters
    ----------
    inner : FeatureProvider
        The provider that actually talks to the encoder — typically
        :class:`~precisionai.agritune.features.provider.OnlineFeatureProvider` or
        :class:`~precisionai.agritune.features.provider.HybridFeatureProvider`. Never a
        :class:`~precisionai.agritune.features.provider.CachedFeatureProvider`, which has no
        network latency to hide.
    batches : Iterable[Sequence[PreparedSample]]
        The exact, ordered sequence of upcoming batches.
    queue_size : int, optional
        How many batches' worth of features to buffer ahead of consumption; also bounds memory
        use, per §19's "use bounded queues so memory does not grow without limit".

    Raises
    ------
    ValueError
        If ``queue_size`` is less than 1.
    """

    def __init__(
        self, inner: FeatureProvider, batches: Iterable[Sequence[PreparedSample]], *, queue_size: int = 2
    ) -> None:
        if queue_size < 1:
            raise ValueError(f"queue_size must be >= 1; got {queue_size}")
        self._inner = inner
        self._batches = iter(batches)
        self._queue: queue.Queue[_QueueItem] = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(target=self._run, daemon=True, name="agritune-feature-prefetch")
        self._started = False
        self._stop_event = threading.Event()

    def get_features(self, samples: Sequence[PreparedSample]) -> EncoderFeatures:
        """Return the next prefetched batch's features, blocking until they are ready.

        Parameters
        ----------
        samples : Sequence[PreparedSample]
            The current batch — must match, in ``sample_id`` and order, the next entry of the
            ``batches`` sequence given at construction time.

        Returns
        -------
        EncoderFeatures

        Raises
        ------
        RuntimeError
            If the background thread's encoding failed, if ``samples`` does not match the next
            expected batch, or if :meth:`get_features` is called more times than ``batches`` had
            entries.
        """
        if not self._started:
            self._started = True
            self._thread.start()

        item = self._queue.get()

        if isinstance(item, _Done):
            raise RuntimeError(
                "PrefetchingFeatureProvider: get_features was called again after every prefetched "
                "batch was already consumed"
            )
        if isinstance(item, _Failure):
            raise RuntimeError("PrefetchingFeatureProvider: background encoding failed") from item.error

        expected_ids = [sample.sample_id for sample in samples]
        if expected_ids != item.sample_ids:
            raise RuntimeError(
                f"PrefetchingFeatureProvider: batch order mismatch — expected {expected_ids}, "
                f"prefetched {item.sample_ids}. The `batches` given at construction must exactly "
                "match the order get_features is called in."
            )
        return item.features

    def close(self) -> None:
        """Stop the background thread and block until it has exited.

        Safe to call whether every batch was consumed (e.g. training finished normally) or
        consumption stopped early (e.g. early stopping, or an epoch cut short) — a thread blocked
        trying to enqueue a batch nobody will ever consume is asked to give up rather than left to
        block :meth:`close` forever.
        """
        if not self._started:
            return
        self._stop_event.set()
        while self._drain_one():
            pass  # unblock a put() that may be waiting on a full queue
        self._thread.join()

    def _drain_one(self) -> bool:
        try:
            self._queue.get_nowait()
        except queue.Empty:
            return False
        return True

    def _run(self) -> None:
        for batch in self._batches:
            if self._stop_event.is_set():
                return
            try:
                features = self._inner.get_features(batch)
            except Exception as exc:  # intentionally broad: forwarded to the consumer, not swallowed
                self._put(_Failure(exc))
                return
            if not self._put(_Success([sample.sample_id for sample in batch], features)):
                return
        self._put(_Done())

    def _put(self, item: _QueueItem) -> bool:
        """Enqueue ``item``, giving up as soon as a stop is requested. Returns whether it was put."""
        while not self._stop_event.is_set():
            if self._try_put(item):
                return True
        return False

    def _try_put(self, item: _QueueItem) -> bool:
        try:
            self._queue.put(item, timeout=0.1)
        except queue.Full:
            return False
        return True
