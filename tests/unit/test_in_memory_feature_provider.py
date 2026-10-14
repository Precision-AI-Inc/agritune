# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.features.memory.InMemoryFeatureProvider."""

from pathlib import Path
from typing import Any

import pytest
import torch

from precisionai.agritune.features.errors import FeatureNotCachedError
from precisionai.agritune.features.memory import InMemoryFeatureProvider
from precisionai.agritune.features.store import DirectoryFeatureStore, ShardedFeatureStore
from precisionai.agritune.schemas.features import EncoderFeatures, concatenate_encoder_features
from precisionai.agritune.schemas.samples import PreparedSample

_CPU = torch.device("cpu")


def _entry(
    value: float,
    *,
    grid: tuple[int, int] = (2, 2),
    patch_dim: int = 3,
    cls_dim: int | None = 2,
    encoder_model: str = "pai-embedding",
    padded_to: int | None = None,
) -> EncoderFeatures:
    num_patches = grid[0] * grid[1]
    tokens = torch.arange(num_patches * patch_dim, dtype=torch.float32).reshape(1, num_patches, patch_dim) + value
    valid = None
    if padded_to is not None:
        tokens = torch.cat([tokens, torch.zeros(1, padded_to - num_patches, patch_dim)], dim=1)
        valid = torch.zeros(1, padded_to, dtype=torch.bool)
        valid[0, :num_patches] = True
    return EncoderFeatures(
        patch_tokens=tokens,
        cls_tokens=torch.full((1, cls_dim), value) if cls_dim is not None else None,
        patch_grid=torch.tensor([list(grid)]),
        valid_patch_mask=valid,
        image_sizes=[(grid[0] * 14, grid[1] * 14)],
        encoder_model=encoder_model,
        encoder_revision=None,
    )


def _sample(sample_id: str) -> PreparedSample:
    return PreparedSample(sample_id=sample_id, image=None, target=None)


def _key(sample: PreparedSample) -> str:
    return sample.sample_id


def _provider(
    store: DirectoryFeatureStore | ShardedFeatureStore, ids: list[str], **kwargs: Any
) -> InMemoryFeatureProvider:
    options: dict[str, Any] = {"storage_device": _CPU, "output_device": _CPU}
    options.update(kwargs)
    return InMemoryFeatureProvider(store, key_fn=_key, samples=[_sample(i) for i in ids], **options)


def _assert_same(actual: EncoderFeatures, expected: EncoderFeatures) -> None:
    assert torch.equal(actual.patch_tokens, expected.patch_tokens)
    assert (actual.cls_tokens is None) == (expected.cls_tokens is None)
    if actual.cls_tokens is not None and expected.cls_tokens is not None:
        assert torch.equal(actual.cls_tokens, expected.cls_tokens)
    assert torch.equal(actual.patch_grid, expected.patch_grid)
    assert (actual.valid_patch_mask is None) == (expected.valid_patch_mask is None)
    if actual.valid_patch_mask is not None and expected.valid_patch_mask is not None:
        assert torch.equal(actual.valid_patch_mask, expected.valid_patch_mask)
    assert actual.image_sizes == expected.image_sizes
    assert actual.encoder_model == expected.encoder_model
    assert actual.encoder_revision == expected.encoder_revision


@pytest.fixture(params=["directory", "sharded"])
def mixed_store(request: pytest.FixtureRequest, tmp_path: Path) -> DirectoryFeatureStore | ShardedFeatureStore:
    """Five entries: four on a 2x2 grid, one on a 2x3 grid (so batches can mix patch counts)."""
    store: DirectoryFeatureStore | ShardedFeatureStore = (
        DirectoryFeatureStore(tmp_path)
        if request.param == "directory"
        else ShardedFeatureStore(tmp_path, entries_per_shard=2)
    )
    for index in range(4):
        store.write(f"s{index}", _entry(float(index * 100)))
    store.write("wide", _entry(900.0, grid=(2, 3)))
    store.flush()
    return store


@pytest.mark.parametrize(
    "batch",
    [
        ["s0", "s1"],  # uniform grids: no mask, no padding
        ["s2", "wide", "s0"],  # mixed grids: padded to 6 with a validity mask
        ["wide"],  # the widest entry alone: no padding needed
        ["s3", "s3"],  # the same sample twice in one batch
    ],
)
def test_matches_concatenating_individual_store_reads(
    mixed_store: DirectoryFeatureStore | ShardedFeatureStore, batch: list[str]
) -> None:
    provider = _provider(mixed_store, ["s0", "s1", "s2", "s3", "wide"])

    actual = provider.get_features([_sample(sample_id) for sample_id in batch])

    expected = concatenate_encoder_features([mixed_store.read(sample_id) for sample_id in batch])
    _assert_same(actual, expected)


def test_entries_stored_with_their_own_padding_are_unpadded_on_load(tmp_path: Path) -> None:
    store = DirectoryFeatureStore(tmp_path)
    store.write("padded", _entry(5.0, grid=(1, 2), padded_to=4))
    store.write("full", _entry(7.0, grid=(2, 2)))

    provider = _provider(store, ["padded", "full"])
    features = provider.get_features([_sample("padded"), _sample("full")])

    assert features.valid_patch_mask is not None
    assert features.valid_patch_mask.tolist() == [[True, True, False, False], [True, True, True, True]]
    assert torch.equal(features.patch_tokens[0, :2], _entry(5.0, grid=(1, 2)).patch_tokens[0])
    assert torch.equal(features.patch_tokens[0, 2:], torch.zeros(2, 3))


def test_entries_without_cls_tokens(tmp_path: Path) -> None:
    store = DirectoryFeatureStore(tmp_path)
    store.write("a", _entry(1.0, cls_dim=None))

    features = _provider(store, ["a"]).get_features([_sample("a")])

    assert features.cls_tokens is None


def test_duplicate_samples_are_loaded_once(mixed_store: DirectoryFeatureStore | ShardedFeatureStore) -> None:
    provider = _provider(mixed_store, ["s0", "s1", "s0", "s1"])

    assert provider.num_rows == 2


def test_float16_storage_returns_float32_tokens_rounded_through_float16(tmp_path: Path) -> None:
    store = DirectoryFeatureStore(tmp_path)
    store.write("a", _entry(0.1234567))

    provider = _provider(store, ["a"], storage_dtype=torch.float16)
    features = provider.get_features([_sample("a")])

    original = store.read("a")
    assert features.patch_tokens.dtype == torch.float32
    assert torch.equal(features.patch_tokens, original.patch_tokens.half().float())
    assert features.cls_tokens is not None
    assert original.cls_tokens is not None
    assert torch.equal(features.cls_tokens, original.cls_tokens.half().float())
    assert provider.nbytes == (4 * 3 + 2) * 2


def test_output_dtype_none_returns_the_storage_dtype(tmp_path: Path) -> None:
    store = DirectoryFeatureStore(tmp_path)
    store.write("a", _entry(1.0))

    features = _provider(store, ["a"], storage_dtype=torch.bfloat16, output_dtype=None).get_features([_sample("a")])

    assert features.patch_tokens.dtype == torch.bfloat16


def test_float16_overflow_is_rejected_while_preloading(tmp_path: Path) -> None:
    store = DirectoryFeatureStore(tmp_path)
    store.write("huge", _entry(1e6))

    with pytest.raises(ValueError, match=r"non-finite patch values after casting to torch\.float16"):
        _provider(store, ["huge"], storage_dtype=torch.float16)


def test_cls_overflow_is_rejected_while_preloading(tmp_path: Path) -> None:
    store = DirectoryFeatureStore(tmp_path)
    entry = _entry(0.0)
    assert entry.cls_tokens is not None
    entry.cls_tokens.fill_(1e6)
    store.write("huge-cls", entry)

    with pytest.raises(ValueError, match="non-finite CLS values"):
        _provider(store, ["huge-cls"], storage_dtype=torch.float16)


def test_unknown_sample_raises_feature_not_cached(mixed_store: DirectoryFeatureStore | ShardedFeatureStore) -> None:
    provider = _provider(mixed_store, ["s0"])

    with pytest.raises(FeatureNotCachedError, match=r"'s1'.*was not preloaded"):
        provider.get_features([_sample("s1")])


def test_empty_request_is_rejected(mixed_store: DirectoryFeatureStore | ShardedFeatureStore) -> None:
    provider = _provider(mixed_store, ["s0"])

    with pytest.raises(ValueError, match="samples must be non-empty"):
        provider.get_features([])


def test_empty_sample_set_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="samples must be non-empty"):
        _provider(DirectoryFeatureStore(tmp_path), [])


def test_sample_missing_from_the_store_fails_at_construction(tmp_path: Path) -> None:
    store = DirectoryFeatureStore(tmp_path)
    store.write("a", _entry(1.0))

    with pytest.raises(KeyError, match="b"):
        _provider(store, ["a", "b"])


def test_mismatched_patch_dimensions_are_rejected(tmp_path: Path) -> None:
    store = DirectoryFeatureStore(tmp_path)
    store.write("a", _entry(1.0, patch_dim=3))
    store.write("b", _entry(1.0, patch_dim=4))

    with pytest.raises(ValueError, match="disagree on embedding dimensions"):
        _provider(store, ["a", "b"])


def test_mismatched_encoder_identity_is_rejected(tmp_path: Path) -> None:
    store = DirectoryFeatureStore(tmp_path)
    store.write("a", _entry(1.0, encoder_model="pai-embedding"))
    store.write("b", _entry(1.0, encoder_model="other-model"))

    with pytest.raises(ValueError, match="disagree on encoder_model"):
        _provider(store, ["a", "b"])


def test_get_features_does_not_rerun_encoder_features_validation(
    mixed_store: DirectoryFeatureStore | ShardedFeatureStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _provider(mixed_store, ["s0", "wide"])
    calls: list[int] = []
    monkeypatch.setattr(EncoderFeatures, "__post_init__", lambda self: calls.append(1))

    provider.get_features([_sample("s0"), _sample("wide")])

    assert calls == []


def test_logs_a_preload_summary(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    store = DirectoryFeatureStore(tmp_path)
    store.write("a", _entry(1.0))

    with caplog.at_level("INFO", logger="agritune"):
        _provider(store, ["a"], show_progress=True)

    assert "preloaded 1 feature entries" in caplog.text


_needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA-enabled machine")


@_needs_cuda
def test_host_storage_with_cuda_output_matches_cpu_results(
    mixed_store: DirectoryFeatureStore | ShardedFeatureStore,
) -> None:
    cuda = torch.device("cuda")
    ids = ["s0", "s1", "s2", "s3", "wide"]
    batch = [_sample("wide"), _sample("s1")]
    on_cpu = _provider(mixed_store, ids).get_features(batch)

    provider = _provider(mixed_store, ids, storage_device=_CPU, output_device=cuda)
    features = provider.get_features(batch)
    torch.cuda.synchronize()

    assert features.patch_tokens.device.type == "cuda"
    _assert_same(features.to("cpu"), on_cpu)


@_needs_cuda
def test_device_storage_keeps_the_buffer_on_the_gpu(mixed_store: DirectoryFeatureStore | ShardedFeatureStore) -> None:
    cuda = torch.device("cuda")
    ids = ["s0", "s1", "s2", "s3", "wide"]
    batch = [_sample("s2"), _sample("wide")]
    on_cpu = _provider(mixed_store, ids).get_features(batch)

    provider = _provider(mixed_store, ids, storage_device=cuda, output_device=cuda, storage_dtype=torch.float16)
    features = provider.get_features(batch)

    assert provider._patch_tokens.device.type == "cuda"
    assert features.patch_tokens.dtype == torch.float32
    torch.testing.assert_close(features.patch_tokens.cpu(), on_cpu.patch_tokens.half().float())
    assert features.valid_patch_mask is not None
    assert on_cpu.valid_patch_mask is not None
    assert torch.equal(features.valid_patch_mask.cpu(), on_cpu.valid_patch_mask)


def test_keys_are_computed_once_per_sample_object_across_epochs(tmp_path: Path) -> None:
    store = DirectoryFeatureStore(tmp_path)
    for index in range(3):
        store.write(f"s{index}", _entry(float(index)))
    samples = [_sample(f"s{index}") for index in range(3)]
    calls: list[str] = []

    def counting_key(sample: PreparedSample) -> str:
        calls.append(sample.sample_id)
        return sample.sample_id

    provider = InMemoryFeatureProvider(
        store, key_fn=counting_key, samples=samples, storage_device=_CPU, output_device=_CPU
    )
    calls.clear()
    for _ in range(3):  # three "epochs" over the same sample objects
        provider.get_features(samples[:2])
        provider.get_features(samples[2:])

    assert sorted(calls) == ["s0", "s1", "s2"]


def test_an_equal_but_distinct_sample_object_is_still_resolved_by_key(tmp_path: Path) -> None:
    store = DirectoryFeatureStore(tmp_path)
    store.write("a", _entry(1.0))
    provider = _provider(store, ["a"])

    first = provider.get_features([_sample("a")])
    second = provider.get_features([_sample("a")])

    assert torch.equal(first.patch_tokens, second.patch_tokens)


def test_preload_spanning_many_copy_chunks_places_every_entry_in_its_row(tmp_path: Path) -> None:
    store = ShardedFeatureStore(tmp_path, entries_per_shard=37)
    ids = [f"s{index}" for index in range(300)]  # several shards and several _COPY_CHUNK-sized chunks
    for index, sample_id in enumerate(ids):
        store.write(sample_id, _entry(float(index)))
    store.flush()

    provider = _provider(store, ids)
    features = provider.get_features([_sample(sample_id) for sample_id in reversed(ids)])

    expected = concatenate_encoder_features([store.read(sample_id) for sample_id in reversed(ids)])
    _assert_same(features, expected)


def test_an_overflow_in_a_later_copy_chunk_still_fails_the_preload(tmp_path: Path) -> None:
    store = DirectoryFeatureStore(tmp_path)
    ids = [f"s{index:03d}" for index in range(200)]
    for index, sample_id in enumerate(ids):
        store.write(sample_id, _entry(1e6 if index == 190 else float(index)))

    with pytest.raises(ValueError, match="non-finite patch values"):
        _provider(store, ids, storage_dtype=torch.float16)
