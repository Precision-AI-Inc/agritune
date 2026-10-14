# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.features.migrate.migrate_store."""

from pathlib import Path

import torch

from precisionai.agritune.features.migrate import MigrationReport, migrate_store
from precisionai.agritune.features.store import DirectoryFeatureStore, ShardedFeatureStore
from precisionai.agritune.schemas.features import EncoderFeatures


def _features(patch_dim: int = 8) -> EncoderFeatures:
    return EncoderFeatures(
        patch_tokens=torch.randn(1, 4, patch_dim),
        cls_tokens=torch.randn(1, 5),
        patch_grid=torch.tensor([[2, 2]]),
        valid_patch_mask=None,
        image_sizes=[(224, 224)],
        encoder_model="pai-embedding",
        encoder_revision=None,
    )


def test_migrate_copies_every_entry_directory_to_sharded(tmp_path: Path) -> None:
    source = DirectoryFeatureStore(tmp_path / "source")
    for key in ("s0", "s1", "s2"):
        source.write(key, _features())

    dest = ShardedFeatureStore(tmp_path / "dest", entries_per_shard=10)
    report = migrate_store(source, dest)

    assert report == MigrationReport(total=3, migrated=3, skipped=0)
    reopened = ShardedFeatureStore(tmp_path / "dest")
    assert sorted(reopened.list_keys()) == ["s0", "s1", "s2"]


def test_migrate_preserves_feature_content(tmp_path: Path) -> None:
    source = DirectoryFeatureStore(tmp_path / "source")
    original = _features()
    source.write("s0", original)

    dest = ShardedFeatureStore(tmp_path / "dest")
    migrate_store(source, dest)

    migrated = dest.read("s0")
    assert migrated.cls_tokens is not None
    assert original.cls_tokens is not None
    assert torch.equal(migrated.patch_tokens, original.patch_tokens)
    assert torch.equal(migrated.cls_tokens, original.cls_tokens)
    assert migrated.encoder_model == original.encoder_model


def test_migrate_is_resumable_and_skips_already_migrated_keys(tmp_path: Path) -> None:
    source = DirectoryFeatureStore(tmp_path / "source")
    for key in ("s0", "s1"):
        source.write(key, _features())

    dest = ShardedFeatureStore(tmp_path / "dest", entries_per_shard=10)
    dest.write("s0", _features())  # simulate a prior interrupted migration that got partway through
    dest.flush()

    report = migrate_store(source, dest)

    assert report == MigrationReport(total=2, migrated=1, skipped=1)


def test_migrate_empty_source_reports_zero_total(tmp_path: Path) -> None:
    source = DirectoryFeatureStore(tmp_path / "source")
    dest = ShardedFeatureStore(tmp_path / "dest")

    report = migrate_store(source, dest)

    assert report == MigrationReport(total=0, migrated=0, skipped=0)


def test_migrate_calls_on_progress_once_per_key_with_running_totals(tmp_path: Path) -> None:
    source = DirectoryFeatureStore(tmp_path / "source")
    for key in ("s0", "s1", "s2"):
        source.write(key, _features())
    dest = ShardedFeatureStore(tmp_path / "dest")

    seen: list[MigrationReport] = []
    migrate_store(source, dest, on_progress=lambda report: seen.append(MigrationReport(**vars(report))))

    assert len(seen) == 3
    assert [snapshot.migrated for snapshot in seen] == [1, 2, 3]
    assert all(snapshot.total == 3 for snapshot in seen)


def test_migrate_sharded_to_directory(tmp_path: Path) -> None:
    source = ShardedFeatureStore(tmp_path / "source", entries_per_shard=10)
    source.write("s0", _features())
    source.flush()

    dest = DirectoryFeatureStore(tmp_path / "dest")
    report = migrate_store(source, dest)

    assert report == MigrationReport(total=1, migrated=1, skipped=0)
    assert dest.has("s0")
