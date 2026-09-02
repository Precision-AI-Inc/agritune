# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.services.config_template_service."""

from pathlib import Path

import pytest
import yaml

from precisionai.agritune.cli.config import load_training_run_config
from precisionai.agritune.services.config_template_service import render_config_template, write_config_template
from precisionai.agritune.services.training_service import TrainingRunConfig


def test_render_config_template_is_valid_yaml_with_every_top_level_field() -> None:
    parsed = yaml.safe_load(render_config_template())
    for key in (
        "manifest_path",
        "feature_store_dir",
        "run_root",
        "run_id",
        "num_classes",
        "encoder_fingerprint",
        "feature_provider",
        "augmentation",
        "feature_augmentation",
        "decoder_name",
        "decoder_kwargs",
        "batch_size",
        "val_fraction",
        "optimizer",
        "scheduler",
        "trainer",
        "loss",
        "val_metric_name",
        "checkpoint_top_k",
        "tracking",
    ):
        assert key in parsed


def test_write_config_template_creates_the_file(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "config.yaml"
    result = write_config_template(output)
    assert result == output
    assert output.read_text(encoding="utf-8") == render_config_template()


def test_write_config_template_refuses_to_overwrite_by_default(tmp_path: Path) -> None:
    output = tmp_path / "config.yaml"
    output.write_text("existing content", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        write_config_template(output)
    assert output.read_text(encoding="utf-8") == "existing content"


def test_write_config_template_overwrites_when_forced(tmp_path: Path) -> None:
    output = tmp_path / "config.yaml"
    output.write_text("existing content", encoding="utf-8")
    write_config_template(output, force=True)
    assert output.read_text(encoding="utf-8") == render_config_template()


def test_generated_template_loads_into_pure_defaults_once_placeholders_are_filled(tmp_path: Path) -> None:
    output = tmp_path / "config.yaml"
    write_config_template(output)
    filled = (
        output.read_text(encoding="utf-8")
        .replace("manifest_path: /path/to/manifest.csv", "manifest_path: manifest.csv")
        .replace("feature_store_dir: /path/to/features", "feature_store_dir: features")
        .replace("run_id: my-run", "run_id: generated-run")
        .replace("num_classes: 2", "num_classes: 4")
    )
    output.write_text(filled, encoding="utf-8")

    config = load_training_run_config(str(output))

    default = TrainingRunConfig(
        manifest_path="manifest.csv",
        feature_store_dir="features",
        run_root="runs",
        run_id="generated-run",
        num_classes=4,
        encoder_fingerprint=config.encoder_fingerprint,
    )
    config.encoder_api_key = default.encoder_api_key = None
    config.original_config = default.original_config = {}
    config.config_overrides = default.config_overrides = []
    # decoder_kwargs is spelled out explicitly (matching mlp_probe's own kwarg defaults) rather
    # than left as TrainingRunConfig's empty-dict default — both are equivalent once forwarded.
    assert config.decoder_kwargs == {"hidden_dims": [], "dropout": 0.0}
    config.decoder_kwargs = default.decoder_kwargs
    assert config == default
