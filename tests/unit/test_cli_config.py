# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.cli.config.load_training_run_config."""

from pathlib import Path

import yaml

from precisionai.agritune.cli.config import load_training_run_config

_BASE_CONFIG = {
    "manifest_path": "manifest.csv",
    "feature_store_dir": "features",
    "run_root": "runs",
    "run_id": "run-1",
    "num_classes": 2,
    "encoder_fingerprint": {"model": "fake-encoder", "revision": "fake-v1", "preprocessing": ""},
}


def _write_config(tmp_path: Path, overrides: dict) -> Path:
    config = {**_BASE_CONFIG, **overrides}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_optimizer_betas_are_converted_to_a_tuple(tmp_path: Path) -> None:
    path = _write_config(tmp_path, {"optimizer": {"name": "adamw", "lr": 0.001, "betas": [0.8, 0.9]}})
    config = load_training_run_config(str(path))
    assert config.optimizer.betas == (0.8, 0.9)


def test_scheduler_is_none_when_omitted(tmp_path: Path) -> None:
    path = _write_config(tmp_path, {})
    config = load_training_run_config(str(path))
    assert config.scheduler is None


def test_scheduler_is_built_when_given(tmp_path: Path) -> None:
    path = _write_config(tmp_path, {"scheduler": {"name": "cosine", "total_steps": 100}})
    config = load_training_run_config(str(path))
    assert config.scheduler is not None
    assert config.scheduler.name == "cosine"
    assert config.scheduler.total_steps == 100


def test_dotlist_overrides_are_applied(tmp_path: Path) -> None:
    path = _write_config(tmp_path, {})
    config = load_training_run_config(str(path), ["num_classes=5", "batch_size=8"])
    assert config.num_classes == 5
    assert config.batch_size == 8


def test_checkpoint_top_k_defaults_to_three(tmp_path: Path) -> None:
    path = _write_config(tmp_path, {})
    config = load_training_run_config(str(path))
    assert config.checkpoint_top_k == 3


def test_checkpoint_top_k_is_configurable(tmp_path: Path) -> None:
    path = _write_config(tmp_path, {"checkpoint_top_k": 1})
    config = load_training_run_config(str(path))
    assert config.checkpoint_top_k == 1
