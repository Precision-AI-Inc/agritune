# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.training.checkpointing.CheckpointManager."""

from pathlib import Path

import pytest
import torch

from precisionai.agritune.training.checkpointing import Checkpoint, CheckpointManager, CheckpointMismatchError
from precisionai.agritune.training.state import TrainingState


def _checkpoint(*, step: int = 0, fingerprints: dict[str, str] | None = None) -> Checkpoint:
    return Checkpoint(
        decoder_state={"weight": [1.0, 2.0]},
        optimizer_state={"lr": 0.01},
        scheduler_state={"last_epoch": 3},
        scaler_state={},
        training_state=TrainingState(epoch=2, micro_step=10, global_optimizer_step=step),
        rng_state={"python": (1, (2,), None)},
        fingerprints=fingerprints or {"encoder": "fake-v1", "config": "abc123"},
    )


def test_has_last_false_before_any_save(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    assert not manager.has_last()


def test_save_then_load_roundtrips_everything(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    original = _checkpoint()
    manager.save(original)

    assert manager.has_last()
    loaded = manager.load(manager.last_path)
    assert loaded.decoder_state == original.decoder_state
    assert loaded.optimizer_state == original.optimizer_state
    assert loaded.scheduler_state == original.scheduler_state
    assert loaded.training_state == original.training_state
    assert loaded.fingerprints == original.fingerprints


def test_is_best_also_writes_best_checkpoint(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    manager.save(_checkpoint(), is_best=True)
    assert manager.best_path.is_file()


def test_periodic_writes_step_numbered_file(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    manager.save(_checkpoint(step=42), periodic=True)
    assert (tmp_path / "step_00000042.ckpt").is_file()


def test_top_k_pruning_keeps_only_best_scores(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, top_k=2)
    for step, metric_value in [(1, 0.5), (2, 0.3), (3, 0.9), (4, 0.1)]:
        manager.save(_checkpoint(step=step), periodic=True, metric_value=metric_value)

    remaining = sorted(p.name for p in tmp_path.glob("step_*.ckpt"))
    # Lower metric_value = better (kept): steps 2 (0.3) and 4 (0.1).
    assert remaining == ["step_00000002.ckpt", "step_00000004.ckpt"]


def test_prune_ignores_ranked_paths_that_are_already_gone(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, top_k=1)
    manager._ranked.append((9.0, tmp_path / "already-gone.ckpt"))
    manager.save(_checkpoint(step=1), periodic=True, metric_value=0.1)

    assert (tmp_path / "step_00000001.ckpt").is_file()
    assert not (tmp_path / "already-gone.ckpt").exists()


def test_prune_removes_multiple_excess_checkpoints_in_one_call(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, top_k=1)
    extras = []
    for step in (1, 2):
        path = tmp_path / f"stale_{step}.ckpt"
        path.write_bytes(b"stale")
        extras.append((float(step), path))
    manager._ranked.extend(extras)
    manager.save(_checkpoint(step=3), periodic=True, metric_value=0.05)

    remaining = list(tmp_path.glob("step_*.ckpt"))
    assert remaining == [tmp_path / "step_00000003.ckpt"]
    assert not (tmp_path / "stale_1.ckpt").exists()
    assert not (tmp_path / "stale_2.ckpt").exists()


def test_top_k_zero_disables_pruning(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, top_k=0)
    for step in range(5):
        manager.save(_checkpoint(step=step), periodic=True, metric_value=float(step))
    assert len(list(tmp_path.glob("step_*.ckpt"))) == 5


def test_periodic_checkpoints_without_metric_are_pruned_by_recency(tmp_path: Path) -> None:
    """Regression test: a mid-epoch checkpoint_every_n_steps save has no validation metric yet
    (metric_value=None) — previously that meant it was never pruned at all, growing without bound.
    It must instead be pruned by recency, independent of the metric-ranked pool."""
    manager = CheckpointManager(tmp_path, top_k=2)
    for step in range(1, 5):
        manager.save(_checkpoint(step=step), periodic=True, metric_value=None)

    remaining = sorted(p.name for p in tmp_path.glob("step_*.ckpt"))
    assert remaining == ["step_00000003.ckpt", "step_00000004.ckpt"]


def test_metric_ranked_and_recency_pools_are_pruned_independently(tmp_path: Path) -> None:
    """A validation-triggered periodic checkpoint (with a metric) and a mid-epoch one (without)
    must not compete for the same top-k slots — each pool is capped at top_k on its own."""
    manager = CheckpointManager(tmp_path, top_k=1)
    manager.save(_checkpoint(step=1), periodic=True, metric_value=None)
    manager.save(_checkpoint(step=2), periodic=True, metric_value=0.5)

    remaining = sorted(p.name for p in tmp_path.glob("step_*.ckpt"))
    assert remaining == ["step_00000001.ckpt", "step_00000002.ckpt"]


def test_top_k_zero_disables_recency_pruning_too(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, top_k=0)
    for step in range(5):
        manager.save(_checkpoint(step=step), periodic=True, metric_value=None)
    assert len(list(tmp_path.glob("step_*.ckpt"))) == 5


def test_load_ignores_unknown_training_state_fields_and_defaults_missing_ones(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    manager.save(_checkpoint())
    payload = torch.load(manager.last_path, map_location="cpu", weights_only=False)
    payload["training_state"].pop("batch_in_epoch", None)
    payload["training_state"].pop("epochs_without_improvement", None)
    payload["training_state"]["legacy_unknown"] = 1
    torch.save(payload, manager.last_path)

    loaded = manager.load(manager.last_path)

    assert loaded.training_state.batch_in_epoch == 0
    assert loaded.training_state.epochs_without_improvement == 0


def test_load_without_expected_fingerprints_skips_check(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    manager.save(_checkpoint(fingerprints={"encoder": "v1"}))
    loaded = manager.load(manager.last_path)  # no expected_fingerprints passed
    assert loaded.fingerprints == {"encoder": "v1"}


def test_load_with_matching_fingerprints_does_not_raise(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    manager.save(_checkpoint(fingerprints={"encoder": "v1"}))
    manager.load(manager.last_path, expected_fingerprints={"encoder": "v1"}, strict=True)


def test_load_with_mismatched_fingerprints_strict_raises(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    manager.save(_checkpoint(fingerprints={"encoder": "v1"}))
    with pytest.raises(CheckpointMismatchError, match="encoder"):
        manager.load(manager.last_path, expected_fingerprints={"encoder": "v2"}, strict=True)


def test_load_with_mismatched_fingerprints_non_strict_warns_and_returns(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    manager.save(_checkpoint(fingerprints={"encoder": "v1"}))
    loaded = manager.load(manager.last_path, expected_fingerprints={"encoder": "v2"}, strict=False)
    assert loaded.fingerprints == {"encoder": "v1"}
