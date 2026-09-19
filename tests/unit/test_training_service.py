# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.services.training_service.

Covers feature_provider selection (cached/online/hybrid), augmentation wiring
(none/offline/online/hybrid), the invalid-combination validation, and tracking backend selection —
on top of the CLI/API end-to-end coverage in test_cli_handlers.py/test_api_run_routes.py, which
only exercises the default (cached, no augmentation, jsonl) path.
"""

import asyncio
from enum import Enum
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from precisionai.agritune.augmentations.feature.pipeline import FeatureAugmentationConfig, FeatureAugmentationPipeline
from precisionai.agritune.augmentations.image.pipeline import AugmentationMode, GeometricConfig
from precisionai.agritune.data.dataset import ManifestDataset
from precisionai.agritune.data.manifest import load_manifest
from precisionai.agritune.data.split import random_split
from precisionai.agritune.encoder.fake import FakeEncoderBackend
from precisionai.agritune.features.keys import EncoderFingerprint
from precisionai.agritune.features.store import DirectoryFeatureStore, ShardedFeatureStore
from precisionai.agritune.optimization.optimizers import OptimizerConfig
from precisionai.agritune.schemas.samples import Sample
from precisionai.agritune.services import training_service
from precisionai.agritune.services.feature_service import build_features
from precisionai.agritune.services.tracking_selection import TrackingSelection
from precisionai.agritune.services.training_service import (
    AugmentationSelection,
    TrainingRunConfig,
    _jsonable,
    run_training,
)
from precisionai.agritune.training.checkpointing import CheckpointMismatchError
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


class _CloseFailingTracker:
    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        pass

    def log_params(self, params: dict[str, object]) -> None:
        pass

    def log_artifact(self, path: str) -> None:
        pass

    def close(self) -> None:
        raise RuntimeError("tracker close failed")


class _Kind(Enum):
    OFFLINE = "offline"


def test_jsonable_serializes_tensors_paths_and_enums() -> None:
    assert _jsonable(torch.tensor([1.0, 2.0])) == [1.0, 2.0]
    assert _jsonable(Path("runs") / "demo") == str(Path("runs") / "demo")
    assert _jsonable(_Kind.OFFLINE) == "offline"


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

    result = run_training(config, store=store, show_progress=True)

    assert result.final_train_state["epoch"] == 1
    assert "mean_iou" in result.val_metrics
    assert "mean_iou" in result.train_metrics

    run_dir = result.run_directory.path
    assert (run_dir / "dataset.json").is_file()
    assert (run_dir / "encoder.json").is_file()
    original = yaml.safe_load((run_dir / "config.original.yaml").read_text(encoding="utf-8"))
    resolved = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
    assert original["decoder_name"] == "mlp_probe"
    assert resolved["decoder_name"] == "mlp_probe"
    assert resolved["optimizer"]["name"] == "adamw"
    run_info = yaml.safe_load((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_info["dataset_fingerprint"]
    assert run_info["split_fingerprint"]
    assert run_info["encoder_fingerprint"]
    assert run_info["feature_cache_fingerprint"]


def test_run_training_cached_provider_does_not_load_images_per_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The image-loading path (``ManifestDataset.__getitem__``) must not scale with dataset size x
    epochs for feature_provider=cached — only the one-time output-size probe may call it."""
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store = DirectoryFeatureStore(tmp_path / "features")
    _precompute(manifest_path, store)

    calls = 0
    original_getitem = ManifestDataset.__getitem__

    def counting_getitem(self: ManifestDataset, index: int) -> Sample:
        nonlocal calls
        calls += 1
        return original_getitem(self, index)

    monkeypatch.setattr(ManifestDataset, "__getitem__", counting_getitem)

    config = _base_config(tmp_path, manifest_path, run_id="cached-no-image-run", trainer=TrainerConfig(max_epochs=2))
    result = run_training(config, store=store)

    assert result.final_train_state["epoch"] == 2
    assert calls == 1  # only the one-time output-size probe — never per-batch, per-epoch


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


def test_feature_augmentation_defaults_to_a_no_op_config(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    config = _base_config(tmp_path, manifest_path, run_id="feature-aug-default-run")

    assert config.feature_augmentation == FeatureAugmentationConfig()


def test_run_training_wires_feature_augmentation_config_into_the_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store = DirectoryFeatureStore(tmp_path / "features")
    _precompute(manifest_path, store)

    captured_configs: list[FeatureAugmentationConfig | None] = []
    real_pipeline_cls = training_service.FeatureAugmentationPipeline

    class _SpyFeatureAugmentationPipeline(real_pipeline_cls):  # type: ignore[misc, valid-type]
        def __init__(self, config: FeatureAugmentationConfig | None = None) -> None:
            captured_configs.append(config)
            super().__init__(config)

    monkeypatch.setattr(training_service, "FeatureAugmentationPipeline", _SpyFeatureAugmentationPipeline)

    config = _base_config(
        tmp_path,
        manifest_path,
        run_id="feature-aug-wiring-run",
        feature_augmentation=FeatureAugmentationConfig(patch_dropout_probability=0.3, gaussian_noise_std=0.02),
    )
    run_training(config, store=store)

    assert len(captured_configs) == 1
    assert captured_configs[0] == FeatureAugmentationConfig(patch_dropout_probability=0.3, gaussian_noise_std=0.02)


def test_run_training_with_strong_feature_augmentation_does_not_crash(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store = DirectoryFeatureStore(tmp_path / "features")
    _precompute(manifest_path, store)

    config = _base_config(
        tmp_path,
        manifest_path,
        run_id="feature-aug-strong-run",
        feature_augmentation=FeatureAugmentationConfig(
            patch_dropout_probability=0.99,  # dropout/cls_dropout/channel_dropout require < 1.0
            token_masking_probability=1.0,
            gaussian_noise_std=5.0,
            cls_dropout_probability=0.99,
            channel_dropout_probability=0.9,
        ),
    )
    result = run_training(config, store=store)

    assert result.final_train_state["epoch"] == 1
    assert "mean_iou" in result.train_metrics
    assert "mean_iou" in result.val_metrics


def test_feature_augmentation_is_applied_to_training_features_but_not_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store = DirectoryFeatureStore(tmp_path / "features")
    _precompute(manifest_path, store)

    apply_calls = []
    real_apply = FeatureAugmentationPipeline.apply

    def _spy_apply(self, features, *, generator=None):  # type: ignore[no-untyped-def]
        apply_calls.append(features)
        return real_apply(self, features, generator=generator)

    monkeypatch.setattr(FeatureAugmentationPipeline, "apply", _spy_apply)

    val_fraction = 0.34
    config = _base_config(
        tmp_path,
        manifest_path,
        run_id="feature-aug-train-only-run",
        feature_augmentation=FeatureAugmentationConfig(patch_dropout_probability=0.5),
        val_fraction=val_fraction,
        batch_size=1,
    )
    run_training(config, store=store)

    rows = load_manifest(str(manifest_path))
    train_fraction = max(1e-6, 1.0 - val_fraction - 1e-6)
    split = random_split(rows, val_fraction=val_fraction, train_fraction=train_fraction, seed=config.seed)

    # One training batch per training sample (batch_size=1), one epoch: feature_augmentation.apply
    # must run exactly once per training batch, and never for the (differently sized) val split.
    assert len(apply_calls) == len(split.train)
    assert len(split.val) > 0
    assert len(split.train) != len(split.val)


def test_run_training_with_resizing_augmentation_does_not_crash_on_target_shape_mismatch(tmp_path: Path) -> None:
    """Regression test: geometric augmentation with resize/random_crop changes training targets'
    spatial size, but validation targets are never augmented — the decoder's one fixed output_size
    must be derived from the augmented shape, and loss/metric computation must tolerate validation
    (or any other) batch not matching it."""
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store = DirectoryFeatureStore(tmp_path / "features")

    config = _base_config(
        tmp_path,
        manifest_path,
        run_id="resize-aug-run",
        feature_provider="online",
        augmentation=AugmentationSelection(
            mode=AugmentationMode.OFFLINE,
            geometric=GeometricConfig(resize=(6, 6), random_crop=(4, 4)),
        ),
    )
    result = run_training(config, store=store)

    assert result.final_train_state["epoch"] == 1
    assert "mean_iou" in result.train_metrics
    assert "mean_iou" in result.val_metrics


def test_offline_precomputed_features_are_compatible_with_cached_training(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store = DirectoryFeatureStore(tmp_path / "features")
    asyncio.run(
        build_features(
            str(manifest_path),
            store=store,
            encoder=FakeEncoderBackend(),
            encoder_fingerprint=_FINGERPRINT,
            augmentation_mode=AugmentationMode.OFFLINE,
        )
    )
    config = _base_config(
        tmp_path,
        manifest_path,
        run_id="offline-cached-run",
        augmentation=AugmentationSelection(mode=AugmentationMode.OFFLINE),
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
        augmentation=AugmentationSelection(mode=AugmentationMode.HYBRID, hybrid_online_probability=1.0),
        trainer=TrainerConfig(max_epochs=2),
    )
    result = run_training(config, store=store)

    assert result.final_train_state["epoch"] == 2
    assert len(store.list_keys()) > len(_ROWS)


def test_hybrid_training_flushes_a_sharded_store_for_reopening(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store_path = tmp_path / "sharded-features"
    store = ShardedFeatureStore(store_path, entries_per_shard=100)
    config = _base_config(tmp_path, manifest_path, run_id="sharded-hybrid-run", feature_provider="hybrid")

    run_training(config, store=store)

    reopened = ShardedFeatureStore(store_path)
    assert reopened.list_keys()


def test_run_training_redacts_encoder_api_key_from_provenance(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store = DirectoryFeatureStore(tmp_path / "features")
    _precompute(manifest_path, store)
    secret = "sk-must-never-be-written"
    config = _base_config(tmp_path, manifest_path, run_id="redacted-run", encoder_api_key=secret)
    config.original_config = {"encoder": {"api_key": secret}}
    config.config_overrides = [f"encoder.api_key={secret}"]

    result = run_training(config, store=store)

    resolved_text = (result.run_directory.path / "config.resolved.yaml").read_text(encoding="utf-8")
    original_text = (result.run_directory.path / "config.original.yaml").read_text(encoding="utf-8")
    assert secret not in resolved_text
    assert secret not in original_text
    assert "<redacted>" in resolved_text
    assert "<redacted>" in original_text


def test_resume_rejects_critical_decoder_config_change(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store = DirectoryFeatureStore(tmp_path / "features")
    _precompute(manifest_path, store)
    run_training(_base_config(tmp_path, manifest_path, run_id="fingerprint-run"), store=store)
    changed = _base_config(
        tmp_path,
        manifest_path,
        run_id="fingerprint-run",
        decoder_kwargs={"hidden_dims": [4]},
        trainer=TrainerConfig(max_epochs=2),
    )

    with pytest.raises(CheckpointMismatchError, match="config"):
        run_training(changed, store=store)


def test_resume_rejects_dataset_content_drift(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store = DirectoryFeatureStore(tmp_path / "features")
    first = _base_config(
        tmp_path,
        manifest_path,
        run_id="dataset-drift-run",
        feature_provider="online",
    )
    run_training(first, store=store)
    changed_image = manifest_path.parent / load_manifest(manifest_path)[0].image_path
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(changed_image)
    resumed = _base_config(
        tmp_path,
        manifest_path,
        run_id="dataset-drift-run",
        feature_provider="online",
        trainer=TrainerConfig(max_epochs=2),
    )

    with pytest.raises(CheckpointMismatchError, match="dataset"):
        run_training(resumed, store=store)


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


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_classes": 0}, "num_classes must be positive"),
        ({"batch_size": 0}, "batch_size must be positive"),
        ({"num_workers": -1}, "num_workers must be non-negative"),
        ({"prefetch_factor": 1}, "prefetch_factor=1 requires num_workers > 0"),
        ({"num_workers": 2, "prefetch_factor": 0}, "prefetch_factor must be positive"),
        ({"store_type": "bogus"}, "unsupported store_type"),
        ({"entries_per_shard": 0}, "entries_per_shard must be positive"),
        ({"feature_read_workers": 0}, "feature_read_workers must be positive"),
        ({"device": "not-a-real-device"}, "invalid device"),
        ({"trainer": TrainerConfig(max_epochs=0)}, "trainer.max_epochs must be positive"),
        ({"checkpoint_top_k": -1}, "checkpoint_top_k must be non-negative"),
    ],
)
def test_run_training_rejects_invalid_numeric_config(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store = DirectoryFeatureStore(tmp_path / "features")
    config = _base_config(tmp_path, manifest_path, run_id="invalid-numeric-run", **overrides)

    with pytest.raises(ValueError, match=message):
        run_training(config, store=store)


@pytest.mark.skipif(torch.cuda.is_available(), reason="requires a CPU-only machine")
def test_run_training_rejects_cuda_device_when_cuda_is_unavailable(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store = DirectoryFeatureStore(tmp_path / "features")
    config = _base_config(tmp_path, manifest_path, run_id="no-cuda-run", device="cuda:0")

    with pytest.raises(ValueError, match="requests CUDA"):
        run_training(config, store=store)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA-enabled machine")
def test_run_training_rejects_out_of_range_cuda_index(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store = DirectoryFeatureStore(tmp_path / "features")
    out_of_range = torch.cuda.device_count()
    config = _base_config(tmp_path, manifest_path, run_id="oor-cuda-run", device=f"cuda:{out_of_range}")

    with pytest.raises(ValueError, match="only has"):
        run_training(config, store=store)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA-enabled machine")
def test_run_training_moves_the_decoder_to_the_configured_cuda_device(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store = DirectoryFeatureStore(tmp_path / "features")
    _precompute(manifest_path, store)
    config = _base_config(
        tmp_path,
        manifest_path,
        run_id="cuda-run",
        feature_provider="cached",
        device="cuda:0",
        trainer=TrainerConfig(max_epochs=1),
    )

    run_training(config, store=store)


def test_run_training_rejects_an_empty_training_split(tmp_path: Path) -> None:
    manifest_path = tmp_path / "empty.csv"
    manifest_path.write_text("sample_id,image_path,mask_path\n", encoding="utf-8")
    store = DirectoryFeatureStore(tmp_path / "features")
    config = _base_config(tmp_path, manifest_path, run_id="empty-run")

    with pytest.raises(ValueError, match="training split is empty"):
        run_training(config, store=store)


def test_tracker_close_failure_does_not_invalidate_completed_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store = DirectoryFeatureStore(tmp_path / "features")
    _precompute(manifest_path, store)
    monkeypatch.setattr(training_service, "build_trackers", lambda *args, **kwargs: _CloseFailingTracker())

    result = run_training(_base_config(tmp_path, manifest_path, run_id="close-fail-run"), store=store)

    assert result.final_train_state["epoch"] == 1
    assert (result.run_directory.path / "run.json").is_file()


def test_run_training_with_no_tracking_backends(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path, rows=_ROWS, image_size=(8, 8))
    store = DirectoryFeatureStore(tmp_path / "features")
    _precompute(manifest_path, store)

    config = _base_config(tmp_path, manifest_path, run_id="no-tracking-run", tracking=TrackingSelection(backends=[]))
    result = run_training(config, store=store)

    assert result.final_train_state["epoch"] == 1


def _build_manifest_with_varying_native_sizes(tmp_path: Path) -> Path:
    """Write a manifest where each sample's own image/mask match, but sizes vary across samples.

    ``build_manifest`` only supports one uniform ``image_size`` for every row (or a single sample
    whose *mask alone* mismatches its own image — a shape albumentations rejects outright, even
    before any resize runs, since it requires an image and its own mask to already agree). Neither
    shape reproduces the real failure mode: different samples natively sized differently, each
    internally consistent — the actual shape a raw agricultural dataset can have.
    """
    lines = ["sample_id,image_path,mask_path,field_id"]
    for index, size in enumerate([(8, 8), (8, 8), (8, 8), (8, 8), (10, 9), (8, 8)]):
        sample_id = f"sample-{index}"
        image_path, mask_path = f"{sample_id}_image.png", f"{sample_id}_mask.png"
        lines.append(f"{sample_id},{image_path},{mask_path},field-a")
        Image.fromarray(np.zeros((size[1], size[0], 3), dtype=np.uint8)).save(tmp_path / image_path)
        Image.fromarray(np.zeros((size[1], size[0]), dtype=np.uint8)).save(tmp_path / mask_path)
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def test_run_training_mode_none_with_non_uniform_native_sizes_requires_resize(tmp_path: Path) -> None:
    """Without geometric.resize, augmentation.mode: none collates a non-uniform-size dataset's
    training batches straight from disk and fails — see
    test_run_training_mode_none_applies_configured_resize for confirmation that setting one now
    fixes this without switching mode."""
    manifest_path = _build_manifest_with_varying_native_sizes(tmp_path)
    store = DirectoryFeatureStore(tmp_path / "features")

    config = _base_config(tmp_path, manifest_path, run_id="nonuniform-none-run", feature_provider="online")
    assert config.augmentation.mode is AugmentationMode.NONE
    assert config.augmentation.geometric.resize is None

    with pytest.raises(RuntimeError, match="stack expects each tensor to be equal size"):
        run_training(config, store=store)


def test_run_training_mode_none_applies_configured_resize(tmp_path: Path) -> None:
    """geometric.resize now applies even under augmentation.mode: none (dimensional normalization,
    not real augmentation — see _build_train_batches), fixing the collate crash above without
    switching mode. Critically, this must not disturb the "none" augmentation fingerprint a feature
    cache built with no --augmentation-config is keyed under, or every cached lookup would miss —
    this exercises exactly that: a feature_provider: cached run against a store precomputed under
    the default (mode: none) fingerprint, on a dataset with non-uniform native mask sizes."""
    manifest_path = _build_manifest_with_varying_native_sizes(tmp_path)
    store = DirectoryFeatureStore(tmp_path / "features")
    _precompute(manifest_path, store)

    config = _base_config(
        tmp_path,
        manifest_path,
        run_id="mode-none-with-resize-run",
        augmentation=AugmentationSelection(mode=AugmentationMode.NONE, geometric=GeometricConfig(resize=(8, 8))),
    )
    result = run_training(config, store=store)

    assert result.final_train_state["epoch"] == 1
