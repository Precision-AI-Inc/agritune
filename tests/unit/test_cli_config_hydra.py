# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Hydra config-group composition path in precisionai.agritune.cli.config.

Exercised against the real packaged ``precisionai/agritune/configs/`` directory — the same files
``agritune train`` ships with — rather than a parallel test fixture, so these tests fail if the
shipped config groups themselves ever break.
"""

from pathlib import Path

import pytest

import precisionai.agritune as agritune_pkg
from precisionai.agritune.augmentations.image.pipeline import AugmentationMode
from precisionai.agritune.cli.config import load_training_run_config

_CONFIG_PATH = Path(agritune_pkg.__file__).parent / "configs" / "config.yaml"


def _overrides(tmp_path: Path, *extra: str) -> list[str]:
    base = [
        "run_id=test-run",
        f"feature_store_dir={tmp_path / 'features'}",
        f"dataset.manifest_path={tmp_path / 'manifest.csv'}",
        "dataset.num_classes=2",
        f"run_root={tmp_path / 'runs'}",
    ]
    return base + list(extra)


def test_packaged_config_is_recognized_as_hydra_composable() -> None:
    assert _CONFIG_PATH.is_file()
    assert "defaults:" in _CONFIG_PATH.read_text()


def test_default_group_selections_resolve(tmp_path: Path) -> None:
    config = load_training_run_config(str(_CONFIG_PATH), _overrides(tmp_path))

    assert config.feature_provider == "cached"
    assert config.augmentation.mode is AugmentationMode.NONE
    assert config.decoder_name == "mlp_probe"
    assert config.optimizer.name == "adamw"
    assert config.scheduler is None
    assert config.tracking.backends == ["jsonl"]
    assert config.encoder_fingerprint.model == "fake-encoder"
    assert config.encoder_fingerprint.revision == "fake-v1"
    assert config.num_classes == 2


def test_group_variant_overrides_swap_the_selection(tmp_path: Path) -> None:
    config = load_training_run_config(
        str(_CONFIG_PATH),
        _overrides(
            tmp_path,
            "decoder=token_fpn",
            "optimizer=sgd",
            "tracking=local",
            "scheduler=cosine",
            "scheduler.total_steps=100",
        ),
    )

    assert config.decoder_name == "token_fpn"
    assert config.optimizer.name == "sgd"
    assert config.tracking.backends == ["jsonl", "tensorboard"]
    assert config.scheduler is not None
    assert config.scheduler.name == "cosine"
    assert config.scheduler.total_steps == 100


def test_decoder_kwargs_are_plumbed_through_from_the_group_config(tmp_path: Path) -> None:
    config = load_training_run_config(str(_CONFIG_PATH), _overrides(tmp_path))
    assert config.decoder_kwargs == {"hidden_dims": []}


def test_decoder_group_variant_overrides_carry_their_own_kwargs(tmp_path: Path) -> None:
    config = load_training_run_config(str(_CONFIG_PATH), _overrides(tmp_path, "decoder=aspp", "decoder.hidden_dim=64"))
    assert config.decoder_name == "aspp"
    assert config.decoder_kwargs["hidden_dim"] == 64
    assert config.decoder_kwargs["atrous_rates"] == [6, 12, 18]


def test_augmentation_offline_group_populates_geometric_and_photometric(tmp_path: Path) -> None:
    config = load_training_run_config(str(_CONFIG_PATH), _overrides(tmp_path, "augmentation=offline"))

    assert config.augmentation.mode is AugmentationMode.OFFLINE
    assert config.augmentation.geometric.horizontal_flip_probability == 0.5
    assert config.augmentation.photometric.brightness_range == (0.8, 1.2)


def test_augmentation_online_requires_a_non_cached_feature_provider_selection(tmp_path: Path) -> None:
    # The config system itself doesn't reject this combination (that's run_training's job — see
    # test_training_service.py); this just confirms both groups resolve independently and
    # correctly so that later validation has accurate values to check.
    config = load_training_run_config(
        str(_CONFIG_PATH), _overrides(tmp_path, "augmentation=online", "feature_provider=cached")
    )
    assert config.augmentation.mode is AugmentationMode.ONLINE
    assert config.feature_provider == "cached"


def test_feature_provider_online_and_hybrid_are_selectable(tmp_path: Path) -> None:
    online = load_training_run_config(str(_CONFIG_PATH), _overrides(tmp_path, "feature_provider=online"))
    hybrid = load_training_run_config(str(_CONFIG_PATH), _overrides(tmp_path, "feature_provider=hybrid"))
    assert online.feature_provider == "online"
    assert hybrid.feature_provider == "hybrid"


def test_remote_encoder_group_requires_base_url(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="base_url"):
        load_training_run_config(str(_CONFIG_PATH), _overrides(tmp_path, "encoder=remote"))


def test_remote_encoder_group_resolves_with_base_url_override(tmp_path: Path) -> None:
    config = load_training_run_config(
        str(_CONFIG_PATH), _overrides(tmp_path, "encoder=remote", "encoder.base_url=https://example.test/v1")
    )
    assert config.encoder_base_url == "https://example.test/v1"
    assert config.encoder_fingerprint.model == "pai-embedding"
    assert config.encoder_fingerprint.revision is None


def test_trainer_max_epochs_override(tmp_path: Path) -> None:
    config = load_training_run_config(str(_CONFIG_PATH), _overrides(tmp_path, "trainer.max_epochs=5"))
    assert config.trainer.max_epochs == 5


def test_scheduler_none_group_disables_the_scheduler(tmp_path: Path) -> None:
    config = load_training_run_config(str(_CONFIG_PATH), _overrides(tmp_path, "scheduler=none"))
    assert config.scheduler is None


def test_manifest_path_missing_raises(tmp_path: Path) -> None:
    overrides = [
        "run_id=test-run",
        f"feature_store_dir={tmp_path / 'features'}",
        "dataset.num_classes=2",
        f"run_root={tmp_path / 'runs'}",
    ]
    with pytest.raises(Exception, match="manifest_path"):
        load_training_run_config(str(_CONFIG_PATH), overrides)
