# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the real (non-stub) ``agritune`` CLI commands, via ``main()``."""

import json
from pathlib import Path

import pytest
import yaml

from precisionai.agritune.cli.main import main
from tests.fixtures.manifest_factory import build_manifest

_ROWS = [
    ("sample-0", "field-a", 0),
    ("sample-1", "field-a", 1),
    ("sample-2", "field-b", 1),
    ("sample-3", "field-b", 0),
    ("sample-4", "field-c", 1),
    ("sample-5", "field-c", 0),
]


def test_dataset_validate_reports_ok_for_clean_manifest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path = build_manifest(tmp_path)
    exit_code = main(["dataset", "validate", "--manifest", str(manifest_path)])
    assert exit_code == 0
    assert "OK" in capsys.readouterr().out


def test_dataset_validate_reports_issues_and_exits_nonzero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path = build_manifest(tmp_path, omit_image_for="sample-1")
    exit_code = main(["dataset", "validate", "--manifest", str(manifest_path)])
    assert exit_code == 1
    assert "missing_image" in capsys.readouterr().err


def test_dataset_inspect_prints_json_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path = build_manifest(tmp_path)
    exit_code = main(["dataset", "inspect", "--manifest", str(manifest_path)])
    assert exit_code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["num_samples"] == 4


def test_features_build_computes_and_reports_stats(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path = build_manifest(tmp_path)
    store_path = tmp_path / "features"

    exit_code = main(["features", "build", "--manifest", str(manifest_path), "--store", str(store_path)])

    assert exit_code == 0
    assert "computed=4" in capsys.readouterr().out
    assert store_path.is_dir()


def test_train_runs_full_pipeline_from_yaml_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store_path = tmp_path / "features"

    build_exit_code = main(["features", "build", "--manifest", str(manifest_path), "--store", str(store_path)])
    assert build_exit_code == 0
    capsys.readouterr()  # discard features-build output

    config = {
        "manifest_path": str(manifest_path),
        "feature_store_dir": str(store_path),
        "run_root": str(tmp_path / "runs"),
        "run_id": "cli-run",
        "num_classes": 2,
        "encoder_fingerprint": {"model": "fake-encoder", "revision": "fake-v1", "preprocessing": ""},
        "decoder_name": "linear",
        "batch_size": 2,
        "val_fraction": 0.34,
        "optimizer": {"name": "adamw", "lr": 0.05},
        "trainer": {"max_epochs": 2},
    }
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(config))

    exit_code = main(["train", "--config", str(config_path)])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "run directory" in out
    assert "final epoch: 2" in out
    assert (tmp_path / "runs" / "cli-run" / "checkpoints" / "last.ckpt").is_file()


def test_train_applies_dotlist_overrides(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store_path = tmp_path / "features"
    main(["features", "build", "--manifest", str(manifest_path), "--store", str(store_path)])
    capsys.readouterr()

    config = {
        "manifest_path": str(manifest_path),
        "feature_store_dir": str(store_path),
        "run_root": str(tmp_path / "runs"),
        "run_id": "cli-run-override",
        "num_classes": 2,
        "encoder_fingerprint": {"model": "fake-encoder", "revision": "fake-v1", "preprocessing": ""},
        "batch_size": 2,
        "val_fraction": 0.34,
        "trainer": {"max_epochs": 1},
    }
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(config))

    exit_code = main(["train", "--config", str(config_path), "trainer.max_epochs=3"])

    assert exit_code == 0
    assert "final epoch: 3" in capsys.readouterr().out
