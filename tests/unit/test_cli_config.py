# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.cli.config.load_training_run_config."""

from pathlib import Path

import yaml

from precisionai.agritune.augmentations.image.pipeline import AugmentationMode
from precisionai.agritune.cli.config import load_augmentation_selection, load_training_run_config

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


def test_load_augmentation_selection_defaults_to_none_mode() -> None:
    selection = load_augmentation_selection(None)
    assert selection.mode is AugmentationMode.NONE


def test_load_augmentation_selection_reads_offline_config(tmp_path: Path) -> None:
    path = tmp_path / "augmentation.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "mode": "offline",
                "variant": 2,
                "geometric": {"horizontal_flip_probability": 0.5, "rotation_max_degrees": 10.0},
                "photometric": {"brightness_range": [0.8, 1.2]},
            }
        )
    )

    selection = load_augmentation_selection(str(path))

    assert selection.mode is AugmentationMode.OFFLINE
    assert selection.variant == 2
    assert selection.geometric.horizontal_flip_probability == 0.5
    assert selection.photometric.brightness_range == (0.8, 1.2)


def test_feature_augmentation_defaults_to_disabled_when_omitted(tmp_path: Path) -> None:
    path = _write_config(tmp_path, {})
    config = load_training_run_config(str(path))
    assert config.feature_augmentation.patch_dropout_probability == 0.0
    assert config.feature_augmentation.gaussian_noise_std == 0.0


def test_feature_augmentation_is_parsed_from_the_config(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        {
            "feature_augmentation": {
                "patch_dropout_probability": 0.1,
                "gaussian_noise_std": 0.02,
                "cls_dropout_probability": 0.05,
            }
        },
    )
    config = load_training_run_config(str(path))
    assert config.feature_augmentation.patch_dropout_probability == 0.1
    assert config.feature_augmentation.gaussian_noise_std == 0.02
    assert config.feature_augmentation.cls_dropout_probability == 0.05
    assert config.feature_augmentation.token_masking_probability == 0.0


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


def test_num_workers_and_pin_memory_default_to_main_process_only(tmp_path: Path) -> None:
    path = _write_config(tmp_path, {})
    config = load_training_run_config(str(path))
    assert config.num_workers == 0
    assert config.pin_memory is False


def test_num_workers_and_pin_memory_are_configurable(tmp_path: Path) -> None:
    path = _write_config(tmp_path, {"num_workers": 4, "pin_memory": True})
    config = load_training_run_config(str(path))
    assert config.num_workers == 4
    assert config.pin_memory is True
