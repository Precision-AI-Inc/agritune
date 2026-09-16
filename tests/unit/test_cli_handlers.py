# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the real (non-stub) ``agritune`` CLI commands, via ``main()``."""

import json
from pathlib import Path

import pytest
import yaml

from precisionai.agritune.cli.main import main
from precisionai.agritune.features.store import DirectoryFeatureStore
from tests.fixtures.manifest_factory import build_manifest

_ROWS = [
    ("sample-0", "field-a", 0),
    ("sample-1", "field-a", 1),
    ("sample-2", "field-b", 1),
    ("sample-3", "field-b", 0),
    ("sample-4", "field-c", 1),
    ("sample-5", "field-c", 0),
]


def test_config_init_writes_a_template_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output_path = tmp_path / "config.yaml"
    exit_code = main(["config", "init", "--output", str(output_path)])
    assert exit_code == 0
    assert output_path.is_file()
    assert "wrote config template" in capsys.readouterr().out


def test_config_init_refuses_to_overwrite_without_force(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output_path = tmp_path / "config.yaml"
    output_path.write_text("existing content")

    exit_code = main(["config", "init", "--output", str(output_path)])

    assert exit_code == 1
    assert "already exists" in capsys.readouterr().err
    assert output_path.read_text() == "existing content"


def test_config_init_overwrites_with_force(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output_path = tmp_path / "config.yaml"
    output_path.write_text("existing content")

    exit_code = main(["config", "init", "--output", str(output_path), "--force"])

    assert exit_code == 0
    assert output_path.read_text() != "existing content"


def test_dataset_init_writes_a_manifest_template_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output_path = tmp_path / "manifest.csv"
    exit_code = main(["dataset", "init", "--output", str(output_path)])
    assert exit_code == 0
    assert output_path.is_file()
    assert "wrote manifest template" in capsys.readouterr().out


def test_dataset_init_refuses_to_overwrite_without_force(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output_path = tmp_path / "manifest.csv"
    output_path.write_text("existing content")

    exit_code = main(["dataset", "init", "--output", str(output_path)])

    assert exit_code == 1
    assert "already exists" in capsys.readouterr().err
    assert output_path.read_text() == "existing content"


def test_dataset_init_overwrites_with_force(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output_path = tmp_path / "manifest.csv"
    output_path.write_text("existing content")

    exit_code = main(["dataset", "init", "--output", str(output_path), "--force"])

    assert exit_code == 0
    assert output_path.read_text() != "existing content"


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


def test_features_build_with_offline_augmentation_also_computes_the_augmented_variant(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = build_manifest(tmp_path)
    store_path = tmp_path / "features"
    augmentation_config_path = tmp_path / "augmentation.yaml"
    augmentation_config_path.write_text(
        yaml.safe_dump({"mode": "offline", "variant": 0, "geometric": {"horizontal_flip_probability": 0.5}})
    )

    exit_code = main(
        [
            "features",
            "build",
            "--manifest",
            str(manifest_path),
            "--store",
            str(store_path),
            "--augmentation-config",
            str(augmentation_config_path),
            "--seed",
            "0",
        ]
    )

    assert exit_code == 0
    assert "computed=8" in capsys.readouterr().out  # 4 samples x (unaugmented + one offline variant)


def test_train_reads_an_offline_augmented_cache_built_via_the_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store_path = tmp_path / "features"
    augmentation = {"mode": "offline", "variant": 0, "geometric": {"horizontal_flip_probability": 0.5}}
    augmentation_config_path = tmp_path / "augmentation.yaml"
    augmentation_config_path.write_text(yaml.safe_dump(augmentation))

    build_exit_code = main(
        [
            "features",
            "build",
            "--manifest",
            str(manifest_path),
            "--store",
            str(store_path),
            "--augmentation-config",
            str(augmentation_config_path),
            "--seed",
            "0",
        ]
    )
    assert build_exit_code == 0
    capsys.readouterr()

    config = {
        "manifest_path": str(manifest_path),
        "feature_store_dir": str(store_path),
        "run_root": str(tmp_path / "runs"),
        "run_id": "offline-aug-run",
        "num_classes": 2,
        "encoder_fingerprint": {"model": "fake-encoder", "revision": "fake-v1", "preprocessing": ""},
        "feature_provider": "cached",
        "augmentation": augmentation,
        "seed": 0,
        "batch_size": 2,
        "val_fraction": 0.34,
        "optimizer": {"name": "adamw", "lr": 0.05},
        "trainer": {"max_epochs": 1},
    }
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(config))

    exit_code = main(["train", "--config", str(config_path)])

    assert exit_code == 0
    assert "final epoch: 1" in capsys.readouterr().out


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
        "decoder_name": "mlp_probe",
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
    assert "train metrics" in out
    assert (tmp_path / "runs" / "cli-run" / "checkpoints" / "last.ckpt").is_file()


def test_train_with_feature_augmentation_config_runs_full_pipeline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store_path = tmp_path / "features"

    build_exit_code = main(["features", "build", "--manifest", str(manifest_path), "--store", str(store_path)])
    assert build_exit_code == 0
    capsys.readouterr()

    config = {
        "manifest_path": str(manifest_path),
        "feature_store_dir": str(store_path),
        "run_root": str(tmp_path / "runs"),
        "run_id": "cli-feature-aug-run",
        "num_classes": 2,
        "encoder_fingerprint": {"model": "fake-encoder", "revision": "fake-v1", "preprocessing": ""},
        "decoder_name": "mlp_probe",
        "batch_size": 2,
        "val_fraction": 0.34,
        "optimizer": {"name": "adamw", "lr": 0.05},
        "trainer": {"max_epochs": 1},
        "feature_augmentation": {
            "patch_dropout_probability": 0.1,
            "gaussian_noise_std": 0.05,
            "channel_dropout_probability": 0.05,
        },
    }
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(config))

    exit_code = main(["train", "--config", str(config_path)])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "final epoch: 1" in out
    assert (tmp_path / "runs" / "cli-feature-aug-run" / "checkpoints" / "last.ckpt").is_file()


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


def _build_features_and_train_via_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> tuple[Path, Path, Path]:
    """Run `features build` then `train` via the real CLI; return (manifest, store, checkpoint)."""
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store_path = tmp_path / "features"
    main(["features", "build", "--manifest", str(manifest_path), "--store", str(store_path)])
    capsys.readouterr()

    config = {
        "manifest_path": str(manifest_path),
        "feature_store_dir": str(store_path),
        "run_root": str(tmp_path / "runs"),
        "run_id": "cli-fixture-run",
        "num_classes": 2,
        "encoder_fingerprint": {"model": "fake-encoder", "revision": "fake-v1", "preprocessing": ""},
        "batch_size": 2,
        "val_fraction": 0.34,
        "optimizer": {"name": "adamw", "lr": 0.05},
        "trainer": {"max_epochs": 1},
    }
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(config))
    main(["train", "--config", str(config_path)])
    capsys.readouterr()

    checkpoint_path = tmp_path / "runs" / "cli-fixture-run" / "checkpoints" / "last.ckpt"
    return manifest_path, store_path, checkpoint_path


def test_features_verify_reports_ok_for_clean_store(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path = build_manifest(tmp_path)
    store_path = tmp_path / "features"
    main(["features", "build", "--manifest", str(manifest_path), "--store", str(store_path)])
    capsys.readouterr()

    exit_code = main(["features", "verify", "--store", str(store_path)])

    assert exit_code == 0
    assert "OK" in capsys.readouterr().out


def test_features_verify_detects_corruption(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path = build_manifest(tmp_path)
    store_path = tmp_path / "features"
    main(["features", "build", "--manifest", str(manifest_path), "--store", str(store_path)])
    capsys.readouterr()

    store = DirectoryFeatureStore(store_path)
    corrupted_key = store.list_keys()[0]
    with (store_path / f"{corrupted_key}.safetensors").open("r+b") as handle:
        handle.seek(0)
        handle.write(b"\x00" * 16)

    exit_code = main(["features", "verify", "--store", str(store_path)])

    assert exit_code == 1
    assert "CORRUPTED" in capsys.readouterr().err


def test_features_inspect_reports_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path = build_manifest(tmp_path)
    store_path = tmp_path / "features"
    main(["features", "build", "--manifest", str(manifest_path), "--store", str(store_path)])
    capsys.readouterr()

    exit_code = main(["features", "inspect", "--store", str(store_path)])

    assert exit_code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["total_entries"] == 4


def test_features_clean_reports_no_removals_for_a_clean_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = build_manifest(tmp_path)
    store_path = tmp_path / "features"
    main(["features", "build", "--manifest", str(manifest_path), "--store", str(store_path)])
    capsys.readouterr()

    exit_code = main(["features", "clean", "--store", str(store_path)])

    assert exit_code == 0
    assert "removed 0 file(s)" in capsys.readouterr().out


def test_features_build_with_sharded_store_type_builds_a_sharded_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = build_manifest(tmp_path)
    store_path = tmp_path / "features"

    exit_code = main(
        [
            "features",
            "build",
            "--manifest",
            str(manifest_path),
            "--store",
            str(store_path),
            "--store-type",
            "sharded",
            "--entries-per-shard",
            "2",
        ]
    )

    assert exit_code == 0
    assert "computed=4" in capsys.readouterr().out
    assert (store_path / "shard_index.json").is_file()  # ShardedFeatureStore, not one file per sample


def test_features_migrate_copies_a_directory_store_into_a_sharded_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = build_manifest(tmp_path)
    source_path = tmp_path / "features"
    main(["features", "build", "--manifest", str(manifest_path), "--store", str(source_path)])
    capsys.readouterr()
    dest_path = tmp_path / "features-sharded"

    exit_code = main(
        [
            "features",
            "migrate",
            "--source",
            str(source_path),
            "--source-type",
            "directory",
            "--dest",
            str(dest_path),
            "--dest-type",
            "sharded",
        ]
    )

    assert exit_code == 0
    assert "migrated=4 skipped=0 total=4" in capsys.readouterr().out
    assert (dest_path / "shard_index.json").is_file()


def test_features_migrate_is_resumable(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path = build_manifest(tmp_path)
    source_path = tmp_path / "features"
    main(["features", "build", "--manifest", str(manifest_path), "--store", str(source_path)])
    capsys.readouterr()
    dest_path = tmp_path / "features-sharded"
    migrate_args = [
        "features",
        "migrate",
        "--source",
        str(source_path),
        "--dest",
        str(dest_path),
        "--dest-type",
        "sharded",
    ]
    main(migrate_args)
    capsys.readouterr()

    exit_code = main(migrate_args)

    assert exit_code == 0
    assert "migrated=0 skipped=4 total=4" in capsys.readouterr().out


def test_evaluate_reports_metrics(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path, store_path, checkpoint_path = _build_features_and_train_via_cli(tmp_path, capsys)

    exit_code = main(
        [
            "evaluate",
            "--manifest",
            str(manifest_path),
            "--store",
            str(store_path),
            "--checkpoint",
            str(checkpoint_path),
            "--num-classes",
            "2",
        ]
    )

    assert exit_code == 0
    result = json.loads(capsys.readouterr().out)
    assert "mean_iou" in result


def test_evaluate_reports_loss_under_requested_loss_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path, store_path, checkpoint_path = _build_features_and_train_via_cli(tmp_path, capsys)

    exit_code = main(
        [
            "evaluate",
            "--manifest",
            str(manifest_path),
            "--store",
            str(store_path),
            "--checkpoint",
            str(checkpoint_path),
            "--num-classes",
            "2",
            "--loss-name",
            "dice",
        ]
    )

    assert exit_code == 0
    result = json.loads(capsys.readouterr().out)
    assert "loss" in result


def test_predict_writes_prediction_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path, store_path, checkpoint_path = _build_features_and_train_via_cli(tmp_path, capsys)
    output_dir = tmp_path / "predictions"

    exit_code = main(
        [
            "predict",
            "--manifest",
            str(manifest_path),
            "--store",
            str(store_path),
            "--checkpoint",
            str(checkpoint_path),
            "--num-classes",
            "2",
            "--output",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert "wrote 6 prediction(s)" in capsys.readouterr().out
    assert len(list(output_dir.glob("*.png"))) == 6


def test_predict_with_overlays_writes_overlay_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path, store_path, checkpoint_path = _build_features_and_train_via_cli(tmp_path, capsys)
    output_dir = tmp_path / "predictions"

    exit_code = main(
        [
            "predict",
            "--manifest",
            str(manifest_path),
            "--store",
            str(store_path),
            "--checkpoint",
            str(checkpoint_path),
            "--num-classes",
            "2",
            "--output",
            str(output_dir),
            "--overlays",
        ]
    )

    assert exit_code == 0
    assert "wrote 12 prediction(s)" in capsys.readouterr().out
    assert len(list(output_dir.glob("*_overlay.png"))) == 6


def test_encoder_benchmark_reports_recommended_settings(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "encoder",
            "benchmark",
            "--batch-sizes",
            "1",
            "2",
            "--concurrencies",
            "1",
            "--num-requests",
            "2",
        ]
    )

    assert exit_code == 0
    assert "Recommended settings" in capsys.readouterr().out
