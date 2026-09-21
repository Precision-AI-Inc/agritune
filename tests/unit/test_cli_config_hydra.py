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
from precisionai.agritune.utils.env import ENCODER_API_KEY_VARIABLE

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
    assert config.original_config["defaults"]
    assert config.config_overrides == _overrides(tmp_path)
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


@pytest.mark.parametrize("variant", ["mlp_probe", "token_fpn", "aspp", "ppm", "segmenter", "mask_former"])
def test_every_decoder_group_composes(tmp_path: Path, variant: str) -> None:
    config = load_training_run_config(str(_CONFIG_PATH), _overrides(tmp_path, f"decoder={variant}"))
    assert config.decoder_name == variant


def test_decoder_kwargs_are_plumbed_through_from_the_group_config(tmp_path: Path) -> None:
    config = load_training_run_config(str(_CONFIG_PATH), _overrides(tmp_path))
    assert config.decoder_kwargs == {"hidden_dims": []}


def test_decoder_group_variant_overrides_carry_their_own_kwargs(tmp_path: Path) -> None:
    config = load_training_run_config(str(_CONFIG_PATH), _overrides(tmp_path, "decoder=aspp", "decoder.hidden_dim=64"))
    assert config.decoder_name == "aspp"
    assert config.decoder_kwargs["hidden_dim"] == 64
    assert config.decoder_kwargs["atrous_rates"] == [6, 12, 18]
    assert config.decoder_kwargs["num_layers"] == 1


def test_ppm_group_kwargs_include_num_layers(tmp_path: Path) -> None:
    config = load_training_run_config(str(_CONFIG_PATH), _overrides(tmp_path, "decoder=ppm"))
    assert config.decoder_kwargs == {"hidden_dim": 128, "pool_sizes": [1, 2, 3, 6], "num_layers": 1}


def test_token_fpn_group_kwargs_include_num_layers(tmp_path: Path) -> None:
    config = load_training_run_config(str(_CONFIG_PATH), _overrides(tmp_path, "decoder=token_fpn"))
    assert config.decoder_kwargs == {"hidden_dim": 128, "num_layers": 2, "cls_fusion": "none"}


def test_token_fpn_group_num_layers_is_overridable(tmp_path: Path) -> None:
    config = load_training_run_config(
        str(_CONFIG_PATH), _overrides(tmp_path, "decoder=token_fpn", "decoder.num_layers=6", "decoder.hidden_dim=256")
    )
    assert config.decoder_kwargs["num_layers"] == 6
    assert config.decoder_kwargs["hidden_dim"] == 256


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


def test_feature_augmentation_defaults_to_none_group(tmp_path: Path) -> None:
    config = load_training_run_config(str(_CONFIG_PATH), _overrides(tmp_path))
    assert config.feature_augmentation.patch_dropout_probability == 0.0
    assert config.feature_augmentation.gaussian_noise_std == 0.0


def test_feature_augmentation_light_group_composes(tmp_path: Path) -> None:
    config = load_training_run_config(str(_CONFIG_PATH), _overrides(tmp_path, "feature_augmentation=light"))
    assert config.feature_augmentation.patch_dropout_probability == 0.05
    assert config.feature_augmentation.gaussian_noise_std == 0.01


def test_feature_augmentation_group_field_overrides(tmp_path: Path) -> None:
    config = load_training_run_config(
        str(_CONFIG_PATH),
        _overrides(tmp_path, "feature_augmentation=light", "feature_augmentation.gaussian_noise_std=0.5"),
    )
    assert config.feature_augmentation.patch_dropout_probability == 0.05
    assert config.feature_augmentation.gaussian_noise_std == 0.5


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


def test_remote_encoder_api_key_resolves_from_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENCODER_API_KEY_VARIABLE, raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(f"{ENCODER_API_KEY_VARIABLE}=sk-from-dotenv\n", encoding="utf-8")

    config = load_training_run_config(
        str(_CONFIG_PATH), _overrides(tmp_path, "encoder=remote", "encoder.base_url=https://example.test/v1")
    )

    assert config.encoder_api_key == "sk-from-dotenv"


@pytest.mark.parametrize("variant", ["adamw", "adam", "sgd"])
def test_every_optimizer_group_composes(tmp_path: Path, variant: str) -> None:
    config = load_training_run_config(str(_CONFIG_PATH), _overrides(tmp_path, f"optimizer={variant}"))
    assert config.optimizer.name == variant


@pytest.mark.parametrize(
    ("variant", "expected_name"),
    [
        ("cosine", "cosine"),
        ("cosine_warmup", "cosine"),
        ("linear_warmup", "linear_warmup"),
        ("polynomial", "polynomial"),
        ("plateau", "plateau"),
    ],
)
def test_every_enabled_scheduler_group_composes(tmp_path: Path, variant: str, expected_name: str) -> None:
    overrides = [f"scheduler={variant}"]
    if variant != "plateau":
        overrides.append("scheduler.total_steps=100")
    config = load_training_run_config(str(_CONFIG_PATH), _overrides(tmp_path, *overrides))

    assert config.scheduler is not None
    assert config.scheduler.name == expected_name


@pytest.mark.parametrize(
    ("variant", "expected_backends"),
    [
        ("null", []),
        ("jsonl", ["jsonl"]),
        ("tensorboard", ["tensorboard"]),
        ("local", ["jsonl", "tensorboard"]),
        ("mlflow", ["mlflow"]),
        ("wandb", ["wandb"]),
        ("comet", ["comet"]),
    ],
)
def test_every_tracking_group_without_required_overrides_composes(
    tmp_path: Path, variant: str, expected_backends: list[str]
) -> None:
    selection = 'tracking="null"' if variant == "null" else f"tracking={variant}"
    config = load_training_run_config(str(_CONFIG_PATH), _overrides(tmp_path, selection))
    assert config.tracking.backends == expected_backends


def test_neptune_tracking_group_composes_with_required_project(tmp_path: Path) -> None:
    config = load_training_run_config(
        str(_CONFIG_PATH), _overrides(tmp_path, "tracking=neptune", "tracking.neptune_project=workspace/project")
    )
    assert config.tracking.backends == ["neptune"]
    assert config.tracking.neptune_project == "workspace/project"


def test_trainer_max_epochs_override(tmp_path: Path) -> None:
    config = load_training_run_config(str(_CONFIG_PATH), _overrides(tmp_path, "trainer.max_epochs=5"))
    assert config.trainer.max_epochs == 5


def test_store_type_defaults_to_directory(tmp_path: Path) -> None:
    config = load_training_run_config(str(_CONFIG_PATH), _overrides(tmp_path))
    assert config.store_type == "directory"
    assert config.entries_per_shard == 1000


def test_store_type_override(tmp_path: Path) -> None:
    config = load_training_run_config(
        str(_CONFIG_PATH), _overrides(tmp_path, "store_type=sharded", "entries_per_shard=250")
    )
    assert config.store_type == "sharded"
    assert config.entries_per_shard == 250


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
