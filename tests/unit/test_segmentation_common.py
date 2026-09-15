# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.services.segmentation_common."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from precisionai.agritune.augmentations.image.pipeline import (
    AugmentationMode,
    AugmentationPipelineConfig,
    GeometricConfig,
    ImageAugmentationPipeline,
    prepare_sample,
)
from precisionai.agritune.data.dataset import ManifestDataset
from precisionai.agritune.encoder.fake import FakeEncoderBackend
from precisionai.agritune.features.keys import EncoderFingerprint
from precisionai.agritune.features.store import DirectoryFeatureStore
from precisionai.agritune.schemas.samples import PreparedSample
from precisionai.agritune.services.feature_service import build_features
from precisionai.agritune.services.segmentation_common import (
    OnlineAugmentedBatches,
    build_cached_feature_provider,
    build_decoder,
    build_image_hash_fn,
    build_prediction_batches,
    build_static_augmented_batches,
    build_training_batches,
    mask_to_target_tensor,
    probe_feature_dims,
)
from precisionai.agritune.tasks.segmentation.decoders.aspp import ASPPDecoder
from precisionai.agritune.tasks.segmentation.decoders.mask_former import MaskFormerDecoder
from precisionai.agritune.tasks.segmentation.decoders.mlp_probe import MLPProbeDecoder
from precisionai.agritune.tasks.segmentation.decoders.pyramid_pooling import PyramidPoolingDecoder
from precisionai.agritune.tasks.segmentation.decoders.segmenter import SegmenterMaskTransformerDecoder
from precisionai.agritune.tasks.segmentation.decoders.token_fpn import CLSFusion, TokenFPNDecoder
from tests.fixtures.image_factory import make_image, make_mask
from tests.fixtures.manifest_factory import build_manifest

_FINGERPRINT = EncoderFingerprint(model="fake-encoder", revision="fake-v1", preprocessing="resize=8x8")


def test_build_decoder_mlp_probe() -> None:
    decoder = build_decoder("mlp_probe", patch_dim=8, cls_dim=None, num_classes=2, output_size=(4, 4))
    assert isinstance(decoder, MLPProbeDecoder)


def test_build_decoder_mlp_probe_forwards_kwargs() -> None:
    decoder = build_decoder(
        "mlp_probe", patch_dim=8, cls_dim=None, num_classes=2, output_size=(4, 4), hidden_dims=(16,)
    )
    assert isinstance(decoder, MLPProbeDecoder)
    assert decoder.hidden_dims == (16,)


def test_build_decoder_token_fpn() -> None:
    decoder = build_decoder("token_fpn", patch_dim=8, cls_dim=None, num_classes=2, output_size=(4, 4))
    assert isinstance(decoder, TokenFPNDecoder)


def test_build_decoder_token_fpn_converts_cls_fusion_string_to_enum() -> None:
    decoder = build_decoder("token_fpn", patch_dim=8, cls_dim=8, num_classes=2, output_size=(4, 4), cls_fusion="concat")
    assert isinstance(decoder, TokenFPNDecoder)
    assert decoder.cls_fusion is CLSFusion.CONCAT


def test_build_decoder_aspp() -> None:
    decoder = build_decoder("aspp", patch_dim=8, cls_dim=None, num_classes=2, output_size=(4, 4))
    assert isinstance(decoder, ASPPDecoder)


def test_build_decoder_ppm() -> None:
    decoder = build_decoder("ppm", patch_dim=8, cls_dim=None, num_classes=2, output_size=(4, 4))
    assert isinstance(decoder, PyramidPoolingDecoder)


def test_build_decoder_segmenter() -> None:
    decoder = build_decoder(
        "segmenter", patch_dim=8, cls_dim=None, num_classes=2, output_size=(4, 4), hidden_dim=8, num_heads=2
    )
    assert isinstance(decoder, SegmenterMaskTransformerDecoder)


def test_build_decoder_mask_former() -> None:
    decoder = build_decoder(
        "mask_former", patch_dim=8, cls_dim=None, num_classes=2, output_size=(4, 4), hidden_dim=8, num_heads=2
    )
    assert isinstance(decoder, MaskFormerDecoder)


def test_build_decoder_unsupported_name_raises() -> None:
    with pytest.raises(ValueError, match="unsupported decoder"):
        build_decoder("unknown", patch_dim=8, cls_dim=None, num_classes=2, output_size=(4, 4))


def test_mask_to_target_tensor_converts_to_long_tensor() -> None:
    tensor = mask_to_target_tensor([[0, 1], [1, 0]])
    assert tensor.dtype == torch.long
    assert tensor.shape == (2, 2)


def test_build_training_batches_chunks_correctly(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path)
    dataset = ManifestDataset(manifest_path)
    batches = build_training_batches(dataset, batch_size=3)
    assert [len(batch.samples) for batch in batches] == [3, 1]


def test_build_training_batches_is_reiterable(tmp_path: Path) -> None:
    """Trainer.fit() walks train/val batches once per epoch — a one-shot generator would silently
    yield nothing from epoch 2 onward, so every pass must independently re-read from disk."""
    manifest_path = build_manifest(tmp_path)
    dataset = ManifestDataset(manifest_path)
    batches = build_training_batches(dataset, batch_size=3)

    first_pass = [len(batch.samples) for batch in batches]
    second_pass = [len(batch.samples) for batch in batches]

    assert first_pass == second_pass == [3, 1]


def test_build_training_batches_supports_multiple_dataloader_workers(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path)
    dataset = ManifestDataset(manifest_path)
    batches = build_training_batches(dataset, batch_size=3, num_workers=2)
    assert [len(batch.samples) for batch in batches] == [3, 1]


def test_build_training_batches_supports_prefetch_factor_with_workers(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path)
    dataset = ManifestDataset(manifest_path)
    batches = build_training_batches(dataset, batch_size=3, num_workers=2, prefetch_factor=1)
    assert [len(batch.samples) for batch in batches] == [3, 1]


def test_build_training_batches_rejects_prefetch_factor_without_workers(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path)
    dataset = ManifestDataset(manifest_path)
    with pytest.raises(ValueError, match="prefetch_factor"):
        list(build_training_batches(dataset, batch_size=3, num_workers=0, prefetch_factor=1))


def test_build_training_batches_pin_memory_does_not_error(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path)
    dataset = ManifestDataset(manifest_path)
    batches = list(build_training_batches(dataset, batch_size=3, pin_memory=True))
    assert [len(batch.samples) for batch in batches] == [3, 1]


def test_build_training_batches_without_resize_rejects_varying_native_sizes(tmp_path: Path) -> None:
    """Reproduces the bug: unequal native mask sizes cannot be torch.stack-ed without a resize."""
    make_image((8, 6)).save(tmp_path / "s0_image.png")
    make_mask((8, 6)).save(tmp_path / "s0_mask.png")
    make_image((10, 7)).save(tmp_path / "s1_image.png")
    make_mask((10, 7)).save(tmp_path / "s1_mask.png")
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "sample_id,image_path,mask_path\ns0,s0_image.png,s0_mask.png\ns1,s1_image.png,s1_mask.png\n"
    )
    dataset = ManifestDataset(manifest_path)

    with pytest.raises(RuntimeError, match="stack expects each tensor to be equal size"):
        list(build_training_batches(dataset, batch_size=2))


def test_build_training_batches_resize_normalizes_varying_native_sizes(tmp_path: Path) -> None:
    make_image((8, 6)).save(tmp_path / "s0_image.png")
    make_mask((8, 6)).save(tmp_path / "s0_mask.png")
    make_image((10, 7)).save(tmp_path / "s1_image.png")
    make_mask((10, 7)).save(tmp_path / "s1_mask.png")
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "sample_id,image_path,mask_path\ns0,s0_image.png,s0_mask.png\ns1,s1_image.png,s1_mask.png\n"
    )
    dataset = ManifestDataset(manifest_path)

    batches = list(build_training_batches(dataset, batch_size=2, resize=(8, 6)))

    assert batches[0].targets.shape == (2, 6, 8)


def test_build_prediction_batches_have_no_targets(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path)
    dataset = ManifestDataset(manifest_path)
    batches = list(build_prediction_batches(dataset, batch_size=2))
    assert [len(batch) for batch in batches] == [2, 2]
    assert all(sample.target is None for batch in batches for sample in batch)


def test_build_prediction_batches_is_lazy(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path)
    dataset = ManifestDataset(manifest_path)
    batches = build_prediction_batches(dataset, batch_size=2)
    assert next(batches) is not None
    assert isinstance(batches, Iterator)


async def test_build_cached_feature_provider_reads_what_was_precomputed(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path)
    store = DirectoryFeatureStore(tmp_path / "features")
    await build_features(
        str(manifest_path), store=store, encoder=FakeEncoderBackend(), encoder_fingerprint=_FINGERPRINT
    )

    provider, rows = build_cached_feature_provider(str(manifest_path), store=store, encoder_fingerprint=_FINGERPRINT)

    assert len(rows) == 4
    sample = PreparedSample(sample_id="sample-0", image=None, target=None)
    features = provider.get_features([sample])
    assert features.batch_size == 1


async def test_probe_feature_dims_reports_patch_and_cls_dims(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path)
    store = DirectoryFeatureStore(tmp_path / "features")
    encoder = FakeEncoderBackend()
    await build_features(str(manifest_path), store=store, encoder=encoder, encoder_fingerprint=_FINGERPRINT)
    provider, _ = build_cached_feature_provider(str(manifest_path), store=store, encoder_fingerprint=_FINGERPRINT)

    patch_dim, cls_dim = probe_feature_dims(provider, PreparedSample(sample_id="sample-0", image=None, target=None))

    assert patch_dim == 384  # FakeEncoderBackend's default patch_dim
    assert cls_dim == 384  # FakeEncoderBackend's default cls_dim


def test_build_image_hash_fn_is_deterministic_and_content_sensitive(tmp_path: Path) -> None:
    # manifest_factory's images are all identical (flat black) by design, so build two manifest
    # rows pointing at genuinely different image content directly, to prove the hash is sensitive
    # to it (not just to the sample_id or manifest row).
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(tmp_path / "s0.png")
    Image.new("RGB", (4, 4), color=(200, 100, 50)).save(tmp_path / "s1.png")
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("sample_id,image_path,mask_path\ns0,s0.png,s0.png\ns1,s1.png,s1.png\n")

    hash_fn = build_image_hash_fn(str(manifest_path))
    sample_0 = PreparedSample(sample_id="s0", image=None, target=None)
    sample_1 = PreparedSample(sample_id="s1", image=None, target=None)

    assert hash_fn(sample_0) == hash_fn(sample_0)
    assert hash_fn(sample_0) != hash_fn(sample_1)  # different images -> different hashes


def _build_gradient_manifest(tmp_path: Path, *, size: tuple[int, int] = (8, 6)) -> Path:
    """A manifest with real (non-flat) images, so geometric/photometric augmentation is visible."""
    for sample_id in ("s0", "s1"):
        make_image(size).save(tmp_path / f"{sample_id}_image.png")
        make_mask(size, fill_value=1).save(tmp_path / f"{sample_id}_mask.png")
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "sample_id,image_path,mask_path\ns0,s0_image.png,s0_mask.png\ns1,s1_image.png,s1_mask.png\n"
    )
    return manifest_path


def test_build_static_augmented_batches_none_mode_matches_build_training_batches(tmp_path: Path) -> None:
    manifest_path = _build_gradient_manifest(tmp_path)
    dataset = ManifestDataset(manifest_path)
    pipeline = ImageAugmentationPipeline()

    plain = list(build_training_batches(dataset, batch_size=2))
    augmented = list(
        build_static_augmented_batches(
            dataset, pipeline=pipeline, mode=AugmentationMode.NONE, global_seed=0, batch_size=2
        )
    )

    assert torch.equal(plain[0].targets, augmented[0].targets)
    for plain_sample, augmented_sample in zip(plain[0].samples, augmented[0].samples, strict=True):
        assert np.array_equal(np.array(plain_sample.image), np.array(augmented_sample.image))


def test_build_static_augmented_batches_offline_applies_augmentation(tmp_path: Path) -> None:
    manifest_path = _build_gradient_manifest(tmp_path)
    dataset = ManifestDataset(manifest_path)
    pipeline = ImageAugmentationPipeline(
        AugmentationPipelineConfig(geometric=GeometricConfig(horizontal_flip_probability=1.0))
    )

    batches = list(
        build_static_augmented_batches(
            dataset, pipeline=pipeline, mode=AugmentationMode.OFFLINE, global_seed=0, batch_size=2, variant=0
        )
    )

    original = dataset[0].image
    flipped = batches[0].samples[0].image
    assert not np.array_equal(np.array(original), np.array(flipped))
    assert batches[0].samples[0].augmentation_metadata is not None


def test_build_static_augmented_batches_offline_is_deterministic(tmp_path: Path) -> None:
    manifest_path = _build_gradient_manifest(tmp_path)
    dataset = ManifestDataset(manifest_path)
    pipeline = ImageAugmentationPipeline(
        AugmentationPipelineConfig(geometric=GeometricConfig(rotation_max_degrees=30.0))
    )

    first = list(
        build_static_augmented_batches(
            dataset, pipeline=pipeline, mode=AugmentationMode.OFFLINE, global_seed=7, batch_size=2, variant=0
        )
    )
    second = list(
        build_static_augmented_batches(
            dataset, pipeline=pipeline, mode=AugmentationMode.OFFLINE, global_seed=7, batch_size=2, variant=0
        )
    )

    for first_sample, second_sample in zip(first[0].samples, second[0].samples, strict=True):
        assert np.array_equal(np.array(first_sample.image), np.array(second_sample.image))


def test_build_static_augmented_batches_is_reiterable(tmp_path: Path) -> None:
    """Same re-iterability requirement as build_training_batches — see that test's docstring."""
    manifest_path = _build_gradient_manifest(tmp_path)
    dataset = ManifestDataset(manifest_path)
    pipeline = ImageAugmentationPipeline(
        AugmentationPipelineConfig(geometric=GeometricConfig(rotation_max_degrees=30.0))
    )
    batches = build_static_augmented_batches(
        dataset, pipeline=pipeline, mode=AugmentationMode.OFFLINE, global_seed=7, batch_size=2, variant=0
    )

    first_pass = list(batches)
    second_pass = list(batches)

    for first_sample, second_sample in zip(first_pass[0].samples, second_pass[0].samples, strict=True):
        assert np.array_equal(np.array(first_sample.image), np.array(second_sample.image))


def test_online_augmented_batches_reaugments_on_each_iteration(tmp_path: Path) -> None:
    manifest_path = _build_gradient_manifest(tmp_path)
    dataset = ManifestDataset(manifest_path)
    pipeline = ImageAugmentationPipeline(
        AugmentationPipelineConfig(geometric=GeometricConfig(rotation_max_degrees=45.0))
    )
    batches = OnlineAugmentedBatches(
        dataset, pipeline=pipeline, mode=AugmentationMode.ONLINE, global_seed=0, batch_size=2
    )

    first_epoch = list(batches)
    second_epoch = list(batches)

    first_sample = first_epoch[0].samples[0]
    second_sample = second_epoch[0].samples[0]
    assert not np.array_equal(np.array(first_sample.image), np.array(second_sample.image))
    assert first_sample.augmentation_metadata is not None
    assert second_sample.augmentation_metadata is not None
    assert first_sample.augmentation_metadata.seed != second_sample.augmentation_metadata.seed


def test_online_augmented_batches_advances_its_epoch_counter(tmp_path: Path) -> None:
    manifest_path = _build_gradient_manifest(tmp_path)
    dataset = ManifestDataset(manifest_path)
    pipeline = ImageAugmentationPipeline()
    batches = OnlineAugmentedBatches(
        dataset, pipeline=pipeline, mode=AugmentationMode.ONLINE, global_seed=0, batch_size=2
    )

    assert batches._epoch == 0
    list(batches)
    assert batches._epoch == 1
    list(batches)
    assert batches._epoch == 2


def test_online_augmented_batches_can_start_at_a_resumed_epoch(tmp_path: Path) -> None:
    manifest_path = _build_gradient_manifest(tmp_path)
    dataset = ManifestDataset(manifest_path)
    batches = OnlineAugmentedBatches(
        dataset,
        pipeline=ImageAugmentationPipeline(),
        mode=AugmentationMode.ONLINE,
        global_seed=0,
        batch_size=2,
    )

    batches.set_epoch(4)
    resumed_epoch = list(batches)

    assert batches._epoch == 5
    assert resumed_epoch[0].samples[0].augmentation_metadata is not None
    expected = prepare_sample(
        dataset[0],
        mode=AugmentationMode.ONLINE,
        pipeline=ImageAugmentationPipeline(),
        global_seed=0,
        epoch=4,
    )
    assert expected.augmentation_metadata is not None
    assert resumed_epoch[0].samples[0].augmentation_metadata.seed == expected.augmentation_metadata.seed


def test_online_augmented_batches_rejects_negative_start_epoch(tmp_path: Path) -> None:
    manifest_path = _build_gradient_manifest(tmp_path)
    batches = OnlineAugmentedBatches(
        ManifestDataset(manifest_path),
        pipeline=ImageAugmentationPipeline(),
        mode=AugmentationMode.ONLINE,
        global_seed=0,
        batch_size=2,
    )
    with pytest.raises(ValueError, match="epoch must be non-negative"):
        batches.set_epoch(-1)


def test_online_augmented_batches_hybrid_mode_is_deterministic_per_epoch(tmp_path: Path) -> None:
    manifest_path = _build_gradient_manifest(tmp_path)
    dataset = ManifestDataset(manifest_path)
    pipeline = ImageAugmentationPipeline(
        AugmentationPipelineConfig(geometric=GeometricConfig(rotation_max_degrees=30.0))
    )

    def _run() -> list:
        batches = OnlineAugmentedBatches(
            dataset,
            pipeline=pipeline,
            mode=AugmentationMode.HYBRID,
            global_seed=3,
            batch_size=2,
            hybrid_online_probability=0.5,
        )
        return [np.array(sample.image) for batch in batches for sample in batch.samples]

    first_run = _run()
    second_run = _run()
    for first_image, second_image in zip(first_run, second_run, strict=True):
        assert np.array_equal(first_image, second_image)  # epoch 0 is deterministic across runs
