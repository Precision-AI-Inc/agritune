# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.services.prediction_service."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from precisionai.agritune.encoder.fake import FakeEncoderBackend, FakeEncoderConfig
from precisionai.agritune.features.keys import EncoderFingerprint
from precisionai.agritune.features.store import DirectoryFeatureStore
from precisionai.agritune.optimization.optimizers import OptimizerConfig
from precisionai.agritune.services.feature_service import build_features
from precisionai.agritune.services.prediction_service import PredictionRunConfig, run_prediction
from precisionai.agritune.services.training_service import TrainingRunConfig, run_training
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
        run_id="predict-fixture-run",
        num_classes=2,
        encoder_fingerprint=_FINGERPRINT,
        batch_size=2,
        val_fraction=0.34,
        optimizer=OptimizerConfig(name="adamw", lr=0.05),
        trainer=TrainerConfig(max_epochs=1),
    )
    result = run_training(config, store=store)
    return manifest_path, result.run_directory.checkpoints_dir / "last.ckpt"


async def test_run_prediction_writes_one_png_per_sample(tmp_path: Path) -> None:
    manifest_path, checkpoint_path = await _train_a_checkpoint(tmp_path)
    store = DirectoryFeatureStore(tmp_path / "features")
    output_dir = tmp_path / "predictions"

    config = PredictionRunConfig(
        manifest_path=str(manifest_path),
        checkpoint_path=str(checkpoint_path),
        output_dir=str(output_dir),
        num_classes=2,
        encoder_fingerprint=_FINGERPRINT,
    )
    written = run_prediction(config, store=store, show_progress=True)

    assert len(written) == 6
    for path in written:
        assert Path(path).is_file()


async def test_run_prediction_output_has_correct_shape_and_labels(tmp_path: Path) -> None:
    manifest_path, checkpoint_path = await _train_a_checkpoint(tmp_path)
    store = DirectoryFeatureStore(tmp_path / "features")
    output_dir = tmp_path / "predictions"

    config = PredictionRunConfig(
        manifest_path=str(manifest_path),
        checkpoint_path=str(checkpoint_path),
        output_dir=str(output_dir),
        num_classes=2,
        encoder_fingerprint=_FINGERPRINT,
    )
    written = run_prediction(config, store=store)

    array = np.array(Image.open(written[0]))
    assert array.shape == (8, 8)
    assert set(np.unique(array).tolist()) <= {0, 1}


async def test_run_prediction_restricts_to_sample_ids(tmp_path: Path) -> None:
    manifest_path, checkpoint_path = await _train_a_checkpoint(tmp_path)
    store = DirectoryFeatureStore(tmp_path / "features")

    config = PredictionRunConfig(
        manifest_path=str(manifest_path),
        checkpoint_path=str(checkpoint_path),
        output_dir=str(tmp_path / "predictions"),
        num_classes=2,
        encoder_fingerprint=_FINGERPRINT,
        sample_ids=["sample-0", "sample-2"],
    )
    written = run_prediction(config, store=store)
    assert len(written) == 2


async def test_run_prediction_writes_overlays_when_requested(tmp_path: Path) -> None:
    manifest_path, checkpoint_path = await _train_a_checkpoint(tmp_path)
    store = DirectoryFeatureStore(tmp_path / "features")
    output_dir = tmp_path / "predictions"

    config = PredictionRunConfig(
        manifest_path=str(manifest_path),
        checkpoint_path=str(checkpoint_path),
        output_dir=str(output_dir),
        num_classes=2,
        encoder_fingerprint=_FINGERPRINT,
        sample_ids=["sample-0"],
        write_overlays=True,
    )
    written = run_prediction(config, store=store)

    assert written == [str(output_dir / "sample-0.png"), str(output_dir / "sample-0_overlay.png")]
    overlay = Image.open(output_dir / "sample-0_overlay.png")
    assert overlay.mode == "RGB"
    assert overlay.size == (8, 8)


async def test_run_prediction_omits_overlays_by_default(tmp_path: Path) -> None:
    manifest_path, checkpoint_path = await _train_a_checkpoint(tmp_path)
    store = DirectoryFeatureStore(tmp_path / "features")
    output_dir = tmp_path / "predictions"

    config = PredictionRunConfig(
        manifest_path=str(manifest_path),
        checkpoint_path=str(checkpoint_path),
        output_dir=str(output_dir),
        num_classes=2,
        encoder_fingerprint=_FINGERPRINT,
        sample_ids=["sample-0"],
    )
    written = run_prediction(config, store=store)

    assert written == [str(output_dir / "sample-0.png")]
    assert not (output_dir / "sample-0_overlay.png").exists()


async def test_run_prediction_empty_sample_set_raises(tmp_path: Path) -> None:
    manifest_path, checkpoint_path = await _train_a_checkpoint(tmp_path)
    store = DirectoryFeatureStore(tmp_path / "features")

    config = PredictionRunConfig(
        manifest_path=str(manifest_path),
        checkpoint_path=str(checkpoint_path),
        output_dir=str(tmp_path / "predictions"),
        num_classes=2,
        encoder_fingerprint=_FINGERPRINT,
        sample_ids=["does-not-exist"],
    )
    with pytest.raises(ValueError, match="no samples to predict"):
        run_prediction(config, store=store)
