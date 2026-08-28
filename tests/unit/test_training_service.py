# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.services.training_service.

Covers feature_provider selection (cached/online/hybrid), augmentation wiring
(none/offline/online/hybrid), the invalid-combination validation, and tracking backend selection —
on top of the CLI/API end-to-end coverage in test_cli_handlers.py/test_api_run_routes.py, which
only exercises the default (cached, no augmentation, jsonl) path.
"""

import asyncio
from pathlib import Path

import pytest

from precisionai.agritune.augmentations.image.pipeline import AugmentationMode, GeometricConfig
from precisionai.agritune.encoder.fake import FakeEncoderBackend
from precisionai.agritune.features.keys import EncoderFingerprint
from precisionai.agritune.features.store import DirectoryFeatureStore
from precisionai.agritune.optimization.optimizers import OptimizerConfig
from precisionai.agritune.services.feature_service import build_features
from precisionai.agritune.services.tracking_selection import TrackingSelection
from precisionai.agritune.services.training_service import AugmentationSelection, TrainingRunConfig, run_training
from precisionai.agritune.training.trainer import TrainerConfig
from tests.fixtures.manifest_factory import build_manifest

_ROWS = [
    ("sample-0", "field-a", 0),
    ("sample-1", "field-a", 1),
    ("sample-2", "field-b", 1),
    ("sample-3", "field-b", 0),
    ("sample-4", "field-c", 1),
    ("sample-5", "field-c", 0),
]
_FINGERPRINT = EncoderFingerprint(model="fake-encoder", revision="fake-v1", preprocessing="")


def _precompute(manifest_path: Path, store: DirectoryFeatureStore) -> None:
    asyncio.run(
        build_features(str(manifest_path), store=store, encoder=FakeEncoderBackend(), encoder_fingerprint=_FINGERPRINT)
    )


def _base_config(tmp_path: Path, manifest_path: Path, *, run_id: str, **overrides: object) -> TrainingRunConfig:
    defaults: dict[str, object] = {
        "manifest_path": str(manifest_path),
        "feature_store_dir": str(tmp_path / "features"),
        "run_root": str(tmp_path / "runs"),
        "run_id": run_id,
        "num_classes": 2,
        "encoder_fingerprint": _FINGERPRINT,
        "batch_size": 2,
        "val_fraction": 0.34,
        "optimizer": OptimizerConfig(name="adamw", lr=0.05),
        "trainer": TrainerConfig(max_epochs=1),
    }
    defaults.update(overrides)
    return TrainingRunConfig(**defaults)  # type: ignore[arg-type]


def test_run_training_default_config_is_backward_compatible(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store = DirectoryFeatureStore(tmp_path / "features")
    _precompute(manifest_path, store)

    config = _base_config(tmp_path, manifest_path, run_id="default-run")
    assert config.feature_provider == "cached"
    assert config.augmentation.mode is AugmentationMode.NONE

    result = run_training(config, store=store)

    assert result.final_train_state["epoch"] == 1
    assert "mean_iou" in result.val_metrics


def test_run_training_with_online_feature_provider_needs_no_precomputed_store(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store = DirectoryFeatureStore(tmp_path / "features")  # deliberately left empty

    config = _base_config(tmp_path, manifest_path, run_id="online-run", feature_provider="online")
    result = run_training(config, store=store)

    assert result.final_train_state["epoch"] == 1


def test_run_training_with_hybrid_feature_provider_writes_through(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store = DirectoryFeatureStore(tmp_path / "features")  # empty; hybrid must fill it in

    config = _base_config(tmp_path, manifest_path, run_id="hybrid-run", feature_provider="hybrid")
    run_training(config, store=store)

    assert list(store.list_keys())  # write-through populated the store during training


def test_run_training_with_offline_augmentation(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store = DirectoryFeatureStore(tmp_path / "features")

    config = _base_config(
        tmp_path,
        manifest_path,
        run_id="offline-aug-run",
        feature_provider="online",
        augmentation=AugmentationSelection(
            mode=AugmentationMode.OFFLINE, geometric=GeometricConfig(horizontal_flip_probability=0.5)
        ),
    )
    result = run_training(config, store=store)

    assert result.final_train_state["epoch"] == 1


def test_run_training_with_online_augmentation_and_online_provider(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store = DirectoryFeatureStore(tmp_path / "features")

    config = _base_config(
        tmp_path,
        manifest_path,
        run_id="online-aug-run",
        feature_provider="online",
        augmentation=AugmentationSelection(mode=AugmentationMode.ONLINE),
        trainer=TrainerConfig(max_epochs=3),
    )
    result = run_training(config, store=store)

    assert result.final_train_state["epoch"] == 3


def test_run_training_with_hybrid_augmentation_and_hybrid_provider(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store = DirectoryFeatureStore(tmp_path / "features")

    config = _base_config(
        tmp_path,
        manifest_path,
        run_id="hybrid-aug-run",
        feature_provider="hybrid",
        augmentation=AugmentationSelection(mode=AugmentationMode.HYBRID, hybrid_online_probability=0.5),
        trainer=TrainerConfig(max_epochs=2),
    )
    result = run_training(config, store=store)

    assert result.final_train_state["epoch"] == 2


def test_run_training_rejects_online_augmentation_with_cached_provider(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store = DirectoryFeatureStore(tmp_path / "features")
    _precompute(manifest_path, store)

    config = _base_config(
        tmp_path,
        manifest_path,
        run_id="invalid-run",
        feature_provider="cached",
        augmentation=AugmentationSelection(mode=AugmentationMode.ONLINE),
    )

    with pytest.raises(ValueError, match="invalid combination"):
        run_training(config, store=store)


def test_run_training_rejects_hybrid_augmentation_with_cached_provider(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store = DirectoryFeatureStore(tmp_path / "features")
    _precompute(manifest_path, store)

    config = _base_config(
        tmp_path,
        manifest_path,
        run_id="invalid-run-2",
        feature_provider="cached",
        augmentation=AugmentationSelection(mode=AugmentationMode.HYBRID),
    )

    with pytest.raises(ValueError, match="invalid combination"):
        run_training(config, store=store)


def test_run_training_rejects_unknown_feature_provider(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store = DirectoryFeatureStore(tmp_path / "features")
    _precompute(manifest_path, store)

    config = _base_config(tmp_path, manifest_path, run_id="invalid-run-3", feature_provider="quantum")

    with pytest.raises(ValueError, match="unsupported feature_provider"):
        run_training(config, store=store)


def test_run_training_with_no_tracking_backends(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store = DirectoryFeatureStore(tmp_path / "features")
    _precompute(manifest_path, store)

    config = _base_config(tmp_path, manifest_path, run_id="no-tracking-run", tracking=TrackingSelection(backends=[]))
    result = run_training(config, store=store)

    assert result.final_train_state["epoch"] == 1
