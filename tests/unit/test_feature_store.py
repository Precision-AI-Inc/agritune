# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.features.store."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

from precisionai.agritune.features import store as store_module
from precisionai.agritune.features.store import DirectoryFeatureStore, ShardedFeatureStore, build_feature_store
from precisionai.agritune.schemas.features import EncoderFeatures


def _single_sample_features(
    *, patch_dim: int = 8, cls_dim: int | None = 5, encoder_model: str = "pai-embedding"
) -> EncoderFeatures:
    return EncoderFeatures(
        patch_tokens=torch.randn(1, 4, patch_dim),
        cls_tokens=torch.randn(1, cls_dim) if cls_dim is not None else None,
        patch_grid=torch.tensor([[2, 2]]),
        valid_patch_mask=None,
        image_sizes=[(224, 224)],
        encoder_model=encoder_model,
        encoder_revision=None,
        metadata={"farm_id": "f1"},
    )


class TestDirectoryFeatureStore:
    def test_has_is_false_before_write(self, tmp_path: Path) -> None:
        store = DirectoryFeatureStore(tmp_path)
        assert not store.has("missing")

    def test_write_then_read_roundtrips(self, tmp_path: Path) -> None:
        store = DirectoryFeatureStore(tmp_path)
        features = _single_sample_features()
        store.write("key1", features)

        assert store.has("key1")
        read_back = store.read("key1")
        assert torch.equal(read_back.patch_tokens, features.patch_tokens)
        assert read_back.cls_tokens is not None
        assert features.cls_tokens is not None
        assert torch.equal(read_back.cls_tokens, features.cls_tokens)
        assert read_back.encoder_model == "pai-embedding"
        assert read_back.metadata == {"farm_id": "f1"}

    def test_read_missing_key_raises(self, tmp_path: Path) -> None:
        store = DirectoryFeatureStore(tmp_path)
        with pytest.raises(KeyError):
            store.read("missing")

    def test_read_summary_missing_key_raises(self, tmp_path: Path) -> None:
        store = DirectoryFeatureStore(tmp_path)
        with pytest.raises(KeyError):
            store.read_summary("missing")

    def test_valid_patch_mask_roundtrips(self, tmp_path: Path) -> None:
        store = DirectoryFeatureStore(tmp_path)
        features = EncoderFeatures(
            patch_tokens=torch.randn(1, 4, 8),
            cls_tokens=None,
            patch_grid=torch.tensor([[1, 3]]),  # 3 real patches out of 4 padded slots
            valid_patch_mask=torch.tensor([[True, True, True, False]]),
            image_sizes=[(224, 224)],
            encoder_model="pai-embedding",
            encoder_revision=None,
        )
        store.write("key1", features)
        read_back = store.read("key1")
        assert read_back.valid_patch_mask is not None
        assert features.valid_patch_mask is not None
        assert torch.equal(read_back.valid_patch_mask, features.valid_patch_mask)

    def test_write_rejects_batch_size_greater_than_one(self, tmp_path: Path) -> None:
        store = DirectoryFeatureStore(tmp_path)
        features = EncoderFeatures(
            patch_tokens=torch.randn(2, 4, 8),
            cls_tokens=None,
            patch_grid=torch.tensor([[2, 2], [2, 2]]),
            valid_patch_mask=None,
            image_sizes=[(224, 224), (224, 224)],
            encoder_model="pai-embedding",
            encoder_revision=None,
        )
        with pytest.raises(ValueError, match="one sample per entry"):
            store.write("key1", features)

    def test_read_summary_does_not_require_reading_tensors(self, tmp_path: Path) -> None:
        store = DirectoryFeatureStore(tmp_path)
        store.write("key1", _single_sample_features(patch_dim=8, cls_dim=5))
        summary = store.read_summary("key1")
        assert summary.patch_dim == 8
        assert summary.cls_dim == 5
        assert summary.num_patches == 4
        assert summary.checksum

    def test_keys_lists_every_written_entry(self, tmp_path: Path) -> None:
        store = DirectoryFeatureStore(tmp_path)
        store.write("key1", _single_sample_features())
        store.write("key2", _single_sample_features())
        assert sorted(store.list_keys()) == ["key1", "key2"]

    def test_persists_across_new_store_instances(self, tmp_path: Path) -> None:
        DirectoryFeatureStore(tmp_path).write("key1", _single_sample_features())
        reopened = DirectoryFeatureStore(tmp_path)
        assert reopened.has("key1")

    def test_flush_is_a_harmless_no_op(self, tmp_path: Path) -> None:
        store = DirectoryFeatureStore(tmp_path)
        store.flush()  # should not raise

    def test_verify_passes_for_an_untouched_entry(self, tmp_path: Path) -> None:
        store = DirectoryFeatureStore(tmp_path)
        store.write("key1", _single_sample_features())
        assert store.verify("key1") is True

    def test_verify_detects_corruption(self, tmp_path: Path) -> None:
        store = DirectoryFeatureStore(tmp_path)
        store.write("key1", _single_sample_features())
        with (tmp_path / "key1.safetensors").open("r+b") as handle:
            handle.seek(0)
            handle.write(b"\x00" * 16)
        assert store.verify("key1") is False

    def test_verify_missing_key_raises(self, tmp_path: Path) -> None:
        store = DirectoryFeatureStore(tmp_path)
        with pytest.raises(KeyError):
            store.verify("missing")

    def test_clean_removes_orphaned_tensor_file(self, tmp_path: Path) -> None:
        store = DirectoryFeatureStore(tmp_path)
        store.write("key1", _single_sample_features())
        (tmp_path / "key1.json").unlink()  # simulate an interrupted write
        removed = store.clean()
        assert removed == ["key1.safetensors"]
        assert not (tmp_path / "key1.safetensors").exists()

    def test_clean_removes_orphaned_meta_file(self, tmp_path: Path) -> None:
        store = DirectoryFeatureStore(tmp_path)
        store.write("key1", _single_sample_features())
        (tmp_path / "key1.safetensors").unlink()  # simulate an interrupted write
        removed = store.clean()
        assert removed == ["key1.json"]
        assert not (tmp_path / "key1.json").exists()

    def test_clean_leaves_intact_pairs_alone(self, tmp_path: Path) -> None:
        store = DirectoryFeatureStore(tmp_path)
        store.write("key1", _single_sample_features())
        assert store.clean() == []
        assert store.has("key1")


class TestShardedFeatureStore:
    def test_rejects_non_positive_entries_per_shard(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="entries_per_shard must be positive"):
            ShardedFeatureStore(tmp_path, entries_per_shard=0)

    def test_write_is_buffered_until_shard_is_full(self, tmp_path: Path) -> None:
        store = ShardedFeatureStore(tmp_path, entries_per_shard=2)
        store.write("key1", _single_sample_features())
        assert store.has("key1")  # visible in-process even though not yet flushed
        assert not (tmp_path / "shard_00000.safetensors").exists()

        store.write("key2", _single_sample_features())  # fills the shard -> auto-flush
        assert (tmp_path / "shard_00000.safetensors").exists()

    def test_read_after_auto_flush_matches_written_features(self, tmp_path: Path) -> None:
        store = ShardedFeatureStore(tmp_path, entries_per_shard=1)
        original = _single_sample_features(patch_dim=6, cls_dim=3)
        store.write("key1", original)

        read_back = store.read("key1")
        assert torch.equal(read_back.patch_tokens, original.patch_tokens)
        assert read_back.cls_tokens is not None
        assert original.cls_tokens is not None
        assert torch.equal(read_back.cls_tokens, original.cls_tokens)
        assert read_back.metadata == {"farm_id": "f1"}

    def test_read_pending_entry_before_flush(self, tmp_path: Path) -> None:
        store = ShardedFeatureStore(tmp_path, entries_per_shard=10)
        original = _single_sample_features()
        store.write("key1", original)
        read_back = store.read("key1")
        assert torch.equal(read_back.patch_tokens, original.patch_tokens)

    def test_read_missing_key_raises(self, tmp_path: Path) -> None:
        store = ShardedFeatureStore(tmp_path)
        with pytest.raises(KeyError):
            store.read("missing")

    def test_concurrent_reads_of_an_uncached_shard_load_it_only_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``CachedFeatureProvider`` reads a batch's keys from a thread pool; without the lock in
        ``_load_shard``, every thread racing to read a not-yet-cached shard would independently
        re-read that same (potentially multi-gigabyte) file, rather than one thread loading it
        while the rest wait and reuse the result."""
        store = ShardedFeatureStore(tmp_path, entries_per_shard=10)
        for index in range(5):
            store.write(f"key{index}", _single_sample_features())
        store.flush()

        real_load_file = store_module.load_file
        call_count = 0
        call_count_lock = threading.Lock()

        def slow_load_file(path: str) -> dict[str, torch.Tensor]:
            nonlocal call_count
            with call_count_lock:
                call_count += 1
            time.sleep(0.05)  # widen the race window so concurrent threads overlap
            return real_load_file(path)

        monkeypatch.setattr(store_module, "load_file", slow_load_file)
        barrier = threading.Barrier(8)

        def read_after_barrier(key: str) -> None:
            barrier.wait()
            store.read(key)

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(read_after_barrier, [f"key{index % 5}" for index in range(8)]))

        assert call_count == 1

    def test_manual_flush_persists_partial_shard(self, tmp_path: Path) -> None:
        store = ShardedFeatureStore(tmp_path, entries_per_shard=10)
        store.write("key1", _single_sample_features())
        store.flush()
        assert (tmp_path / "shard_00000.safetensors").exists()

    def test_flush_with_nothing_pending_is_a_no_op(self, tmp_path: Path) -> None:
        store = ShardedFeatureStore(tmp_path)
        store.flush()
        assert store.list_keys() == []

    def test_multiple_shards_get_distinct_names(self, tmp_path: Path) -> None:
        store = ShardedFeatureStore(tmp_path, entries_per_shard=1)
        store.write("key1", _single_sample_features())
        store.write("key2", _single_sample_features())
        assert (tmp_path / "shard_00000.safetensors").exists()
        assert (tmp_path / "shard_00001.safetensors").exists()

    def test_index_persists_across_new_store_instances(self, tmp_path: Path) -> None:
        first = ShardedFeatureStore(tmp_path, entries_per_shard=1)
        first.write("key1", _single_sample_features())

        reopened = ShardedFeatureStore(tmp_path)
        assert reopened.has("key1")
        assert reopened.read("key1").encoder_model == "pai-embedding"

    def test_context_manager_flushes_on_exit(self, tmp_path: Path) -> None:
        with ShardedFeatureStore(tmp_path, entries_per_shard=10) as store:
            store.write("key1", _single_sample_features())
        assert (tmp_path / "shard_00000.safetensors").exists()

    def test_read_summary_for_pending_entry(self, tmp_path: Path) -> None:
        store = ShardedFeatureStore(tmp_path, entries_per_shard=10)
        store.write("key1", _single_sample_features(patch_dim=8, cls_dim=5))
        summary = store.read_summary("key1")
        assert summary.patch_dim == 8
        assert summary.checksum == ""

    def test_read_summary_for_flushed_entry(self, tmp_path: Path) -> None:
        store = ShardedFeatureStore(tmp_path, entries_per_shard=1)
        store.write("key1", _single_sample_features(patch_dim=8, cls_dim=5))
        summary = store.read_summary("key1")
        assert summary.checksum

    def test_read_summary_missing_key_raises(self, tmp_path: Path) -> None:
        store = ShardedFeatureStore(tmp_path)
        with pytest.raises(KeyError):
            store.read_summary("missing")

    def test_keys_reflects_only_flushed_entries(self, tmp_path: Path) -> None:
        store = ShardedFeatureStore(tmp_path, entries_per_shard=10)
        store.write("key1", _single_sample_features())
        assert store.list_keys() == []  # not yet flushed
        store.flush()
        assert store.list_keys() == ["key1"]

    def test_features_without_cls_or_mask_roundtrip(self, tmp_path: Path) -> None:
        store = ShardedFeatureStore(tmp_path, entries_per_shard=1)
        features = _single_sample_features(cls_dim=None)
        store.write("key1", features)
        read_back = store.read("key1")
        assert read_back.cls_tokens is None

    def test_verify_passes_for_an_untouched_flushed_entry(self, tmp_path: Path) -> None:
        store = ShardedFeatureStore(tmp_path, entries_per_shard=1)
        store.write("key1", _single_sample_features())
        assert store.verify("key1") is True

    def test_verify_detects_corruption(self, tmp_path: Path) -> None:
        store = ShardedFeatureStore(tmp_path, entries_per_shard=1)
        store.write("key1", _single_sample_features())
        shard_path = tmp_path / "shard_00000.safetensors"
        with shard_path.open("r+b") as handle:
            handle.seek(0)
            handle.write(b"\x00" * 16)
        assert store.verify("key1") is False

    def test_verify_pending_entry_raises_value_error(self, tmp_path: Path) -> None:
        store = ShardedFeatureStore(tmp_path, entries_per_shard=10)
        store.write("key1", _single_sample_features())
        with pytest.raises(ValueError, match="not been flushed"):
            store.verify("key1")

    def test_verify_missing_key_raises_key_error(self, tmp_path: Path) -> None:
        store = ShardedFeatureStore(tmp_path)
        with pytest.raises(KeyError):
            store.verify("missing")

    def test_clean_removes_unreferenced_shard(self, tmp_path: Path) -> None:
        store = ShardedFeatureStore(tmp_path, entries_per_shard=1)
        store.write("key1", _single_sample_features())
        (tmp_path / "shard_00099.safetensors").write_bytes(b"orphan")  # simulate a stray shard
        removed = store.clean()
        assert removed == ["shard_00099.safetensors"]
        assert not (tmp_path / "shard_00099.safetensors").exists()
        assert store.has("key1")

    def test_clean_leaves_referenced_shards_alone(self, tmp_path: Path) -> None:
        store = ShardedFeatureStore(tmp_path, entries_per_shard=1)
        store.write("key1", _single_sample_features())
        assert store.clean() == []
        assert (tmp_path / "shard_00000.safetensors").exists()

    def test_context_manager_flushes_pending_entries_on_exit(self, tmp_path: Path) -> None:
        with ShardedFeatureStore(tmp_path, entries_per_shard=10) as store:
            store.write("key1", _single_sample_features())
            assert not (tmp_path / "shard_00000.safetensors").exists()  # not yet flushed
        assert (tmp_path / "shard_00000.safetensors").exists()
        assert ShardedFeatureStore(tmp_path).has("key1")


class TestBuildFeatureStore:
    def test_directory_type_builds_a_directory_store(self, tmp_path: Path) -> None:
        store = build_feature_store("directory", tmp_path)
        assert isinstance(store, DirectoryFeatureStore)

    def test_sharded_type_builds_a_sharded_store(self, tmp_path: Path) -> None:
        store = build_feature_store("sharded", tmp_path, entries_per_shard=50)
        assert isinstance(store, ShardedFeatureStore)
        store.write("key1", _single_sample_features())
        store.flush()
        assert ShardedFeatureStore(tmp_path).list_keys() == ["key1"]

    def test_unsupported_type_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unsupported store_type: 'bogus'"):
            build_feature_store("bogus", tmp_path)
