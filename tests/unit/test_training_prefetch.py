# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.training.prefetch.PrefetchingFeatureLoader."""

import threading
from collections.abc import Sequence

import pytest
import torch

from precisionai.agritune.schemas.features import EncoderFeatures
from precisionai.agritune.schemas.samples import PreparedSample
from precisionai.agritune.training.batch import TrainingBatch
from precisionai.agritune.training.prefetch import PrefetchingFeatureLoader
from tests.fixtures.fake_pipeline import FakeFeatureProvider


def _sample(sample_id: str) -> PreparedSample:
    return PreparedSample(sample_id=sample_id, image=None, target=None)


def _batch(sample_id: str, num_pixels: int = 4) -> TrainingBatch:
    return TrainingBatch(samples=[_sample(sample_id)], targets=torch.zeros(1, num_pixels, dtype=torch.long))


def test_yields_every_batch_in_order() -> None:
    provider = FakeFeatureProvider(patch_dim=4, cls_dim=None, patch_grid=(2, 2))
    batches = [_batch("a"), _batch("b"), _batch("c")]

    loader = PrefetchingFeatureLoader(batches, provider, device=torch.device("cpu"), non_blocking=False)
    results = list(loader)

    assert [batch.samples[0].sample_id for batch, _, _ in results] == ["a", "b", "c"]


def test_features_and_targets_match_what_the_provider_and_batch_would_return_directly() -> None:
    provider = FakeFeatureProvider(patch_dim=4, cls_dim=None, patch_grid=(2, 2))
    batch = _batch("a")

    loader = PrefetchingFeatureLoader([batch], provider, device=torch.device("cpu"), non_blocking=False)
    _, features, targets = next(loader)

    expected_features = provider.get_features(batch.samples)
    assert torch.equal(features.patch_tokens, expected_features.patch_tokens)
    assert torch.equal(targets, batch.targets)


def test_empty_batches_raises_stop_iteration_immediately() -> None:
    provider = FakeFeatureProvider(patch_dim=4, cls_dim=None, patch_grid=(2, 2))

    loader = PrefetchingFeatureLoader([], provider, device=torch.device("cpu"), non_blocking=False)

    with pytest.raises(StopIteration):
        next(loader)


def test_moves_features_and_targets_to_the_given_device(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_devices: list[torch.device] = []
    original_to = EncoderFeatures.to

    def spy_to(self: EncoderFeatures, device: torch.device | str, *, non_blocking: bool = False) -> EncoderFeatures:
        seen_devices.append(torch.device(device))
        return original_to(self, device, non_blocking=non_blocking)

    monkeypatch.setattr(EncoderFeatures, "to", spy_to)
    provider = FakeFeatureProvider(patch_dim=4, cls_dim=None, patch_grid=(2, 2))

    loader = PrefetchingFeatureLoader([_batch("a")], provider, device=torch.device("cpu"), non_blocking=False)
    next(loader)

    assert seen_devices == [torch.device("cpu")]


def test_fetches_the_next_batch_while_the_current_one_is_being_consumed() -> None:
    # A real disk/network read can't be observed directly in a unit test, but the same overlap
    # this class exists for shows up as: the *second* batch's fetch is already done by the time
    # the caller asks for it, having run concurrently with whatever the caller was doing between
    # the two next() calls.
    started = threading.Event()
    release = threading.Event()

    class _BlockingProvider:
        def get_features(self, samples: Sequence[PreparedSample]) -> EncoderFeatures:
            if samples[0].sample_id == "b":
                started.set()
                release.wait(timeout=5)
            return FakeFeatureProvider(patch_dim=4, cls_dim=None, patch_grid=(2, 2)).get_features(samples)

    loader = PrefetchingFeatureLoader(
        [_batch("a"), _batch("b")], _BlockingProvider(), device=torch.device("cpu"), non_blocking=False
    )
    next(loader)  # consumes "a"; submits the fetch for "b" in the background

    assert started.wait(timeout=5)  # "b"'s fetch is already underway, unprompted by a next() call for it
    release.set()
    batch, _, _ = next(loader)
    assert batch.samples[0].sample_id == "b"


def test_propagates_an_exception_raised_by_the_feature_provider() -> None:
    class _FailingProvider:
        def get_features(self, samples: Sequence[PreparedSample]) -> EncoderFeatures:
            raise RuntimeError("boom")

    loader = PrefetchingFeatureLoader([_batch("a")], _FailingProvider(), device=torch.device("cpu"), non_blocking=False)

    with pytest.raises(RuntimeError, match="boom"):
        next(loader)


def test_is_its_own_iterator() -> None:
    provider = FakeFeatureProvider(patch_dim=4, cls_dim=None, patch_grid=(2, 2))
    loader = PrefetchingFeatureLoader([_batch("a")], provider, device=torch.device("cpu"), non_blocking=False)
    assert iter(loader) is loader


def test_context_manager_shuts_down_executor_when_abandoned_before_exhaustion() -> None:
    """Regression test: without __enter__/__exit__, a consumer loop abandoned early (e.g. via an
    exception raised elsewhere in the loop body) never reached __next__'s own StopIteration-time
    shutdown, leaking the background thread. Using the loader as a context manager must free it
    regardless of how the ``with`` block exits."""
    provider = FakeFeatureProvider(patch_dim=4, cls_dim=None, patch_grid=(2, 2))
    loader = PrefetchingFeatureLoader(
        [_batch("a"), _batch("b")], provider, device=torch.device("cpu"), non_blocking=False
    )

    with loader:
        next(loader)  # only the first of two batches consumed; loop abandoned early

    assert loader._executor._shutdown


def test_context_manager_shuts_down_executor_when_the_body_raises() -> None:
    provider = FakeFeatureProvider(patch_dim=4, cls_dim=None, patch_grid=(2, 2))
    loader = PrefetchingFeatureLoader([_batch("a")], provider, device=torch.device("cpu"), non_blocking=False)

    with pytest.raises(RuntimeError, match="boom"), loader:
        raise RuntimeError("boom")

    assert loader._executor._shutdown
