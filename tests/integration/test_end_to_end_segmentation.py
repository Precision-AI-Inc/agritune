# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""The main end-to-end integration test:

tiny dataset -> fake encoder -> feature precompute -> segmentation decoder -> train ->
checkpoint -> resume -> evaluate.

No external resources are used (only ``FakeEncoderBackend``), so this runs in default CI — it is
not marked ``@pytest.mark.integration`` (reserved for tests that need a real network resource).
"""

from pathlib import Path

from precisionai.agritune.encoder.fake import FakeEncoderBackend, FakeEncoderConfig
from precisionai.agritune.features.keys import EncoderFingerprint
from precisionai.agritune.features.store import DirectoryFeatureStore
from precisionai.agritune.optimization.optimizers import OptimizerConfig
from precisionai.agritune.services.evaluation_service import EvaluationRunConfig, run_evaluation
from precisionai.agritune.services.feature_service import build_features
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


async def test_full_pipeline_train_checkpoint_resume_evaluate(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))

    store = DirectoryFeatureStore(tmp_path / "features")
    encoder = FakeEncoderBackend(FakeEncoderConfig(patch_dim=8, cls_dim=None, patch_grid=(2, 2)))
    stats = await build_features(str(manifest_path), store=store, encoder=encoder, encoder_fingerprint=_FINGERPRINT)
    assert stats.computed == len(_ROWS)
    assert stats.failed == 0

    run_root = tmp_path / "runs"
    base_config = TrainingRunConfig(
        manifest_path=str(manifest_path),
        feature_store_dir=str(tmp_path / "features"),
        run_root=str(run_root),
        run_id="e2e-run",
        num_classes=2,
        encoder_fingerprint=_FINGERPRINT,
        decoder_name="mlp_probe",
        batch_size=2,
        val_fraction=0.34,
        seed=0,
        optimizer=OptimizerConfig(name="adamw", lr=0.05),
        trainer=TrainerConfig(max_epochs=2),
    )

    first_result = run_training(base_config, store=store)

    assert "mean_iou" in first_result.val_metrics
    assert first_result.final_train_state["epoch"] == 2
    run_dir = first_result.run_directory
    assert (run_dir.path / "config.resolved.yaml").is_file()
    assert (run_dir.path / "run.json").is_file()
    assert (run_dir.path / "environment.json").is_file()
    assert (run_dir.path / "git.json").is_file()
    assert (run_dir.checkpoints_dir / "last.ckpt").is_file()
    assert (run_dir.metrics_path).is_file()

    # Resume: same run_root/run_id (so CheckpointManager sees the existing last.ckpt), more epochs.
    resume_config = TrainingRunConfig(
        manifest_path=base_config.manifest_path,
        feature_store_dir=base_config.feature_store_dir,
        run_root=base_config.run_root,
        run_id=base_config.run_id,
        num_classes=2,
        encoder_fingerprint=_FINGERPRINT,
        decoder_name="mlp_probe",
        batch_size=2,
        val_fraction=0.34,
        seed=0,
        optimizer=OptimizerConfig(name="adamw", lr=0.05),
        trainer=TrainerConfig(max_epochs=4),
    )
    second_result = run_training(resume_config, store=store)

    assert second_result.final_train_state["epoch"] == 4
    assert (
        second_result.final_train_state["global_optimizer_step"]
        > first_result.final_train_state["global_optimizer_step"]
    )

    evaluation_metrics = run_evaluation(
        EvaluationRunConfig(
            manifest_path=str(manifest_path),
            checkpoint_path=str(run_dir.checkpoints_dir / "last.ckpt"),
            num_classes=2,
            encoder_fingerprint=_FINGERPRINT,
            batch_size=2,
        ),
        store=store,
    )
    assert "mean_iou" in evaluation_metrics
    assert "pixel_accuracy" in evaluation_metrics
