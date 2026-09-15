# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.services.evaluation_service."""

from pathlib import Path

import pytest

from precisionai.agritune.augmentations.image.pipeline import (
    AugmentationMode,
    AugmentationPipelineConfig,
    GeometricConfig,
    ImageAugmentationPipeline,
)
from precisionai.agritune.encoder.fake import FakeEncoderBackend, FakeEncoderConfig
from precisionai.agritune.features.errors import FeatureNotCachedError
from precisionai.agritune.features.keys import EncoderFingerprint
from precisionai.agritune.features.store import DirectoryFeatureStore
from precisionai.agritune.optimization.optimizers import OptimizerConfig
from precisionai.agritune.services.evaluation_service import EvaluationRunConfig, run_evaluation
from precisionai.agritune.services.feature_service import build_features
from precisionai.agritune.services.training_service import TrainingRunConfig, run_training
from precisionai.agritune.tasks.segmentation.losses import SegmentationLossConfig
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
_FINGERPRINT = EncoderFingerprint(model="fake-encoder", revision="fake-v1", preprocessing="resize=8x8")


async def _train_a_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store = DirectoryFeatureStore(tmp_path / "features")
    encoder = FakeEncoderBackend(FakeEncoderConfig(patch_dim=8, cls_dim=None, patch_grid=(2, 2)))
    await build_features(str(manifest_path), store=store, encoder=encoder, encoder_fingerprint=_FINGERPRINT)

    config = TrainingRunConfig(
        manifest_path=str(manifest_path),
        feature_store_dir=str(tmp_path / "features"),
        run_root=str(tmp_path / "runs"),
        run_id="eval-fixture-run",
        num_classes=2,
        encoder_fingerprint=_FINGERPRINT,
        batch_size=2,
        val_fraction=0.34,
        optimizer=OptimizerConfig(name="adamw", lr=0.05),
        trainer=TrainerConfig(max_epochs=1),
    )
    result = run_training(config, store=store)
    return manifest_path, result.run_directory.checkpoints_dir / "last.ckpt"


async def test_run_evaluation_reports_metrics(tmp_path: Path) -> None:
    manifest_path, checkpoint_path = await _train_a_checkpoint(tmp_path)
    store = DirectoryFeatureStore(tmp_path / "features")

    config = EvaluationRunConfig(
        manifest_path=str(manifest_path),
        checkpoint_path=str(checkpoint_path),
        num_classes=2,
        encoder_fingerprint=_FINGERPRINT,
    )
    metrics = run_evaluation(config, store=store, show_progress=True)

    assert "mean_iou" in metrics
    assert "pixel_accuracy" in metrics


async def test_run_evaluation_restricts_to_sample_ids(tmp_path: Path) -> None:
    manifest_path, checkpoint_path = await _train_a_checkpoint(tmp_path)
    store = DirectoryFeatureStore(tmp_path / "features")

    config = EvaluationRunConfig(
        manifest_path=str(manifest_path),
        checkpoint_path=str(checkpoint_path),
        num_classes=2,
        encoder_fingerprint=_FINGERPRINT,
        sample_ids=["sample-0", "sample-1"],
    )
    metrics = run_evaluation(config, store=store)
    assert "mean_iou" in metrics


async def test_run_evaluation_reports_loss_under_default_ce_config(tmp_path: Path) -> None:
    manifest_path, checkpoint_path = await _train_a_checkpoint(tmp_path)
    store = DirectoryFeatureStore(tmp_path / "features")

    config = EvaluationRunConfig(
        manifest_path=str(manifest_path),
        checkpoint_path=str(checkpoint_path),
        num_classes=2,
        encoder_fingerprint=_FINGERPRINT,
    )
    metrics = run_evaluation(config, store=store)

    assert "loss" in metrics
    assert metrics["loss"] >= 0.0


async def test_run_evaluation_loss_config_changes_reported_loss(tmp_path: Path) -> None:
    manifest_path, checkpoint_path = await _train_a_checkpoint(tmp_path)
    store = DirectoryFeatureStore(tmp_path / "features")

    ce_config = EvaluationRunConfig(
        manifest_path=str(manifest_path),
        checkpoint_path=str(checkpoint_path),
        num_classes=2,
        encoder_fingerprint=_FINGERPRINT,
        loss=SegmentationLossConfig(name="ce"),
    )
    dice_config = EvaluationRunConfig(
        manifest_path=str(manifest_path),
        checkpoint_path=str(checkpoint_path),
        num_classes=2,
        encoder_fingerprint=_FINGERPRINT,
        loss=SegmentationLossConfig(name="dice"),
    )

    ce_loss = run_evaluation(ce_config, store=store)["loss"]
    dice_loss = run_evaluation(dice_config, store=store)["loss"]

    assert ce_loss != pytest.approx(dice_loss)


async def test_run_evaluation_offline_mode_requires_matching_augmented_cache(tmp_path: Path) -> None:
    """A checkpoint trained on offline-augmented features must not silently fall back to native
    (unaugmented) cached features at evaluation time — see evaluation_service's augmentation_mode."""
    manifest_path, checkpoint_path = await _train_a_checkpoint(tmp_path)
    store = DirectoryFeatureStore(tmp_path / "features")
    pipeline = ImageAugmentationPipeline(AugmentationPipelineConfig(geometric=GeometricConfig(resize=(4, 4))))

    config = EvaluationRunConfig(
        manifest_path=str(manifest_path),
        checkpoint_path=str(checkpoint_path),
        num_classes=2,
        encoder_fingerprint=_FINGERPRINT,
        sample_ids=["sample-0"],
        augmentation_mode=AugmentationMode.OFFLINE,
        augmentation_pipeline=pipeline,
    )
    with pytest.raises(FeatureNotCachedError):
        run_evaluation(config, store=store)


async def test_run_evaluation_offline_mode_reads_matching_augmented_cache(tmp_path: Path) -> None:
    manifest_path, checkpoint_path = await _train_a_checkpoint(tmp_path)
    store = DirectoryFeatureStore(tmp_path / "features")
    encoder = FakeEncoderBackend(FakeEncoderConfig(patch_dim=8, cls_dim=None, patch_grid=(2, 2)))
    pipeline = ImageAugmentationPipeline(AugmentationPipelineConfig(geometric=GeometricConfig(resize=(4, 4))))
    await build_features(
        str(manifest_path),
        store=store,
        encoder=encoder,
        encoder_fingerprint=_FINGERPRINT,
        augmentation_mode=AugmentationMode.OFFLINE,
        augmentation_pipeline=pipeline,
    )

    config = EvaluationRunConfig(
        manifest_path=str(manifest_path),
        checkpoint_path=str(checkpoint_path),
        num_classes=2,
        encoder_fingerprint=_FINGERPRINT,
        sample_ids=["sample-0"],
        augmentation_mode=AugmentationMode.OFFLINE,
        augmentation_pipeline=pipeline,
    )
    metrics = run_evaluation(config, store=store)

    assert "mean_iou" in metrics


async def test_run_evaluation_rejects_online_augmentation_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="augmentation_mode"):
        EvaluationRunConfig(
            manifest_path=str(tmp_path / "manifest.csv"),
            checkpoint_path=str(tmp_path / "last.ckpt"),
            num_classes=2,
            encoder_fingerprint=_FINGERPRINT,
            augmentation_mode=AugmentationMode.ONLINE,
        )


async def test_run_evaluation_empty_sample_set_raises(tmp_path: Path) -> None:
    manifest_path, checkpoint_path = await _train_a_checkpoint(tmp_path)
    store = DirectoryFeatureStore(tmp_path / "features")

    config = EvaluationRunConfig(
        manifest_path=str(manifest_path),
        checkpoint_path=str(checkpoint_path),
        num_classes=2,
        encoder_fingerprint=_FINGERPRINT,
        sample_ids=["does-not-exist"],
    )
    with pytest.raises(ValueError, match="no samples to evaluate"):
        run_evaluation(config, store=store)
