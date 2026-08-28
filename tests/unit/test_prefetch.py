# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.features.prefetch.PrefetchingFeatureProvider."""

import time
from collections.abc import Sequence

import pytest
import torch

from precisionai.agritune.features.prefetch import PrefetchingFeatureProvider, _Success
from precisionai.agritune.schemas.features import EncoderFeatures
from precisionai.agritune.schemas.protocols import FeatureProvider
from precisionai.agritune.schemas.samples import PreparedSample


def _tag(sample_id: str) -> EncoderFeatures:
    """A minimal, valid EncoderFeatures tagged with which sample it stands in for."""
    return EncoderFeatures(
        patch_tokens=torch.zeros(1, 1, 1),
        cls_tokens=None,
        patch_grid=torch.tensor([[1, 1]]),
        valid_patch_mask=None,
        image_sizes=[(1, 1)],
        encoder_model="spy",
        encoder_revision=None,
        metadata={"tag": f"features-for-{sample_id}"},
    )


class _SpyProvider:
    """Records every call and returns a features stand-in tagging which batch it was."""

    def __init__(self, *, latency_seconds: float = 0.0, fail_on_batch: int | None = None) -> None:
        self.call_count = 0
        self._latency_seconds = latency_seconds
        self._fail_on_batch = fail_on_batch

    def get_features(self, samples: Sequence[PreparedSample]) -> EncoderFeatures:
        self.call_count += 1
        if self._fail_on_batch is not None and self.call_count == self._fail_on_batch:
            raise ValueError("simulated encoder failure")
        if self._latency_seconds:
            time.sleep(self._latency_seconds)
        return _tag(samples[0].sample_id)


def _batch(sample_id: str) -> list[PreparedSample]:
    return [PreparedSample(sample_id=sample_id, image=None, target=None)]


def test_get_features_returns_batches_in_order() -> None:
    batches = [_batch("s0"), _batch("s1"), _batch("s2")]
    provider = PrefetchingFeatureProvider(_SpyProvider(), batches)

    results = [provider.get_features(batch) for batch in batches]

    assert [result.metadata["tag"] for result in results] == [
        "features-for-s0",
        "features-for-s1",
        "features-for-s2",
    ]
    provider.close()


def test_prefetching_runs_ahead_of_consumption() -> None:
    spy = _SpyProvider(latency_seconds=0.05)
    batches = [_batch(f"s{i}") for i in range(5)]
    provider = PrefetchingFeatureProvider(spy, batches, queue_size=2)

    provider.get_features(batches[0])  # blocks until the first batch is ready, starting the thread
    time.sleep(0.15)  # let the background thread get ahead without us asking for more

    assert spy.call_count >= 2  # it kept encoding without another get_features() call
    provider.close()


def test_get_features_raises_when_batch_order_does_not_match() -> None:
    provider = PrefetchingFeatureProvider(_SpyProvider(), [_batch("s0")])

    with pytest.raises(RuntimeError, match="batch order mismatch"):
        provider.get_features(_batch("wrong-sample"))

    provider.close()


def test_get_features_raises_once_every_batch_is_consumed() -> None:
    batches = [_batch("s0")]
    provider = PrefetchingFeatureProvider(_SpyProvider(), batches)
    provider.get_features(batches[0])

    with pytest.raises(RuntimeError, match="called again after every prefetched batch"):
        provider.get_features(_batch("s0"))

    provider.close()


def test_inner_provider_error_propagates() -> None:
    batches = [_batch("s0"), _batch("s1")]
    provider = PrefetchingFeatureProvider(_SpyProvider(fail_on_batch=1), batches)

    with pytest.raises(RuntimeError, match="background encoding failed") as exc_info:
        provider.get_features(batches[0])
    assert isinstance(exc_info.value.__cause__, ValueError)

    provider.close()


def test_queue_size_must_be_at_least_one() -> None:
    with pytest.raises(ValueError, match="queue_size must be >= 1"):
        PrefetchingFeatureProvider(_SpyProvider(), [_batch("s0")], queue_size=0)


def test_close_joins_the_background_thread_without_hanging() -> None:
    batches = [_batch("s0"), _batch("s1")]
    provider = PrefetchingFeatureProvider(_SpyProvider(), batches)
    for batch in batches:
        provider.get_features(batch)

    provider.close()  # must return promptly; the thread has already produced its _Done sentinel


def test_close_before_starting_is_a_no_op() -> None:
    provider = PrefetchingFeatureProvider(_SpyProvider(), [_batch("s0")])
    provider.close()  # never started (get_features was never called) — must not hang or raise


def test_try_put_returns_false_when_the_queue_is_full() -> None:
    # Exercises _try_put/_run's backpressure path directly from the main thread — coverage.py
    # does not trace the background thread by default, so this is verified here instead of by
    # actually running PrefetchingFeatureProvider's own thread.
    provider = PrefetchingFeatureProvider(_SpyProvider(), [_batch("s0")], queue_size=1)
    already_queued = _Success(["already-queued"], _tag("already-queued"))
    provider._queue.put_nowait(already_queued)  # white-box: fill the queue directly

    assert provider._try_put(_Success(["s1"], _tag("s1"))) is False

    provider.close()


def test_run_exits_immediately_once_stopped() -> None:
    # Same rationale as above: _run() itself only executes on the background thread in normal
    # use, so call it directly, synchronously, after pre-setting the stop flag.
    provider = PrefetchingFeatureProvider(_SpyProvider(), [_batch("s0")])
    provider._stop_event.set()

    provider._run()  # must return immediately, without touching the queue or the inner provider

    assert provider._queue.empty()


def test_prefetching_feature_provider_satisfies_protocol() -> None:
    provider = PrefetchingFeatureProvider(_SpyProvider(), [_batch("s0")])
    assert isinstance(provider, FeatureProvider)
    provider.close()
