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
    PhotometricConfig,
    prepare_sample,
)
from precisionai.agritune.data.dataset import ManifestDataset
from precisionai.agritune.encoder.fake import FakeEncoderBackend
from precisionai.agritune.features.keys import EncoderFingerprint, hash_augmentation
from precisionai.agritune.features.provider import _MAX_READ_WORKERS
from precisionai.agritune.features.store import DirectoryFeatureStore
from precisionai.agritune.schemas.samples import PreparedSample
from precisionai.agritune.services import segmentation_common
from precisionai.agritune.services.feature_service import build_features
from precisionai.agritune.services.segmentation_common import (
    OnlineAugmentedBatches,
    PreloadedBatches,
    build_cached_feature_provider,
    build_decoder,
    build_image_hash_fn,
    build_prediction_batches,
    build_static_augmented_batches,
    build_training_batches,
    mask_to_target_tensor,
    probe_feature_dims,
    validate_device,
)
from precisionai.agritune.tasks.segmentation.decoders.aspp import ASPPDecoder
from precisionai.agritune.tasks.segmentation.decoders.mask_former import MaskFormerDecoder
from precisionai.agritune.tasks.segmentation.decoders.mlp_probe import MLPProbeDecoder
from precisionai.agritune.tasks.segmentation.decoders.pyramid_pooling import PyramidPoolingDecoder
from precisionai.agritune.tasks.segmentation.decoders.segmenter import SegmenterMaskTransformerDecoder
from precisionai.agritune.tasks.segmentation.decoders.token_fpn import CLSFusion, TokenFPNDecoder
from precisionai.agritune.training.batch import TrainingBatch
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


def test_build_prediction_batches_resize_normalizes_image_size(tmp_path: Path) -> None:
    """Regression test: build_prediction_batches previously had no resize parameter at all, so
    predicting against a checkpoint trained with a resize had no way to match it — see
    test_build_training_batches_resize_normalizes_varying_native_sizes for the training-side
    equivalent this mirrors."""
    make_image((8, 6)).save(tmp_path / "s0_image.png")
    make_mask((8, 6)).save(tmp_path / "s0_mask.png")
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("sample_id,image_path,mask_path\ns0,s0_image.png,s0_mask.png\n")
    dataset = ManifestDataset(manifest_path)

    batches = list(build_prediction_batches(dataset, batch_size=1, resize=(4, 3)))

    assert batches[0][0].image.size == (4, 3)


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


async def test_build_cached_feature_provider_forwards_max_read_workers(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path)
    store = DirectoryFeatureStore(tmp_path / "features")
    await build_features(
        str(manifest_path), store=store, encoder=FakeEncoderBackend(), encoder_fingerprint=_FINGERPRINT
    )

    provider, _ = build_cached_feature_provider(
        str(manifest_path), store=store, encoder_fingerprint=_FINGERPRINT, max_read_workers=4
    )

    assert provider._max_read_workers == 4


async def test_build_cached_feature_provider_defaults_max_read_workers_when_omitted(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path)
    store = DirectoryFeatureStore(tmp_path / "features")
    await build_features(
        str(manifest_path), store=store, encoder=FakeEncoderBackend(), encoder_fingerprint=_FINGERPRINT
    )

    provider, _ = build_cached_feature_provider(str(manifest_path), store=store, encoder_fingerprint=_FINGERPRINT)

    assert provider._max_read_workers == _MAX_READ_WORKERS


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


def test_build_static_augmented_batches_load_images_false_never_sets_image(tmp_path: Path) -> None:
    manifest_path = _build_gradient_manifest(tmp_path)
    dataset = ManifestDataset(manifest_path)
    pipeline = ImageAugmentationPipeline(
        AugmentationPipelineConfig(geometric=GeometricConfig(horizontal_flip_probability=1.0))
    )

    batches = list(
        build_static_augmented_batches(
            dataset,
            pipeline=pipeline,
            mode=AugmentationMode.OFFLINE,
            global_seed=0,
            batch_size=2,
            load_images=False,
        )
    )

    for sample in batches[0].samples:
        assert sample.image is None
        assert sample.augmentation_metadata is not None


def test_build_static_augmented_batches_load_images_false_matches_load_images_true_targets(tmp_path: Path) -> None:
    manifest_path = _build_gradient_manifest(tmp_path)
    dataset = ManifestDataset(manifest_path)
    pipeline = ImageAugmentationPipeline(
        AugmentationPipelineConfig(
            geometric=GeometricConfig(horizontal_flip_probability=1.0, rotation_max_degrees=15.0),
            photometric=PhotometricConfig(brightness_range=(0.6, 1.4), blur_probability=1.0),
        )
    )

    with_images = list(
        build_static_augmented_batches(
            dataset, pipeline=pipeline, mode=AugmentationMode.OFFLINE, global_seed=3, batch_size=2, variant=1
        )
    )
    without_images = list(
        build_static_augmented_batches(
            dataset,
            pipeline=pipeline,
            mode=AugmentationMode.OFFLINE,
            global_seed=3,
            batch_size=2,
            variant=1,
            load_images=False,
        )
    )

    assert torch.equal(with_images[0].targets, without_images[0].targets)
    for with_sample, without_sample in zip(with_images[0].samples, without_images[0].samples, strict=True):
        assert hash_augmentation(with_sample.augmentation_metadata) == hash_augmentation(
            without_sample.augmentation_metadata
        )


async def test_build_static_augmented_batches_load_images_false_still_hits_the_cache(tmp_path: Path) -> None:
    """The end-to-end proof: features cached from the real image+mask pipeline must still be found
    by a provider fed samples built by the image-free (mask-only) batch-building path — even with
    both geometric and photometric randomness active, mirroring a real offline flip/rotate/color
    config."""
    manifest_path = _build_gradient_manifest(tmp_path)
    store = DirectoryFeatureStore(tmp_path / "features")
    pipeline = ImageAugmentationPipeline(
        AugmentationPipelineConfig(
            geometric=GeometricConfig(horizontal_flip_probability=1.0, rotation_max_degrees=20.0),
            photometric=PhotometricConfig(contrast_range=(0.5, 1.5), noise_probability=1.0),
        )
    )
    await build_features(
        str(manifest_path),
        store=store,
        encoder=FakeEncoderBackend(),
        encoder_fingerprint=_FINGERPRINT,
        augmentation_mode=AugmentationMode.OFFLINE,
        augmentation_pipeline=pipeline,
        global_seed=9,
        augmentation_variant=0,
    )

    dataset = ManifestDataset(manifest_path)
    provider, _ = build_cached_feature_provider(str(manifest_path), store=store, encoder_fingerprint=_FINGERPRINT)
    batches = list(
        build_static_augmented_batches(
            dataset,
            pipeline=pipeline,
            mode=AugmentationMode.OFFLINE,
            global_seed=9,
            batch_size=2,
            variant=0,
            load_images=False,
        )
    )

    features = provider.get_features(batches[0].samples)
    assert features.batch_size == 2


def test_build_training_batches_load_images_false_never_sets_image(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path)
    dataset = ManifestDataset(manifest_path)

    batches = list(build_training_batches(dataset, batch_size=2, resize=(4, 3), load_images=False))

    for sample in batches[0].samples:
        assert sample.image is None
    assert batches[0].targets.shape[1:] == (3, 4)


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


def test_validate_device_accepts_cpu() -> None:
    assert validate_device("cpu") == torch.device("cpu")


def test_validate_device_rejects_invalid_device_string() -> None:
    with pytest.raises(ValueError, match="invalid device"):
        validate_device("not-a-real-device")


@pytest.mark.skipif(torch.cuda.is_available(), reason="requires a CPU-only machine")
def test_validate_device_rejects_cuda_when_unavailable() -> None:
    with pytest.raises(ValueError, match="requests CUDA"):
        validate_device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA-enabled machine")
def test_validate_device_rejects_out_of_range_cuda_index() -> None:
    out_of_range = torch.cuda.device_count()
    with pytest.raises(ValueError, match="only has"):
        validate_device(f"cuda:{out_of_range}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA-enabled machine")
def test_validate_device_accepts_a_valid_cuda_device() -> None:
    assert validate_device("cuda:0") == torch.device("cuda:0")


def test_build_image_hash_fn_reads_each_image_only_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path = build_manifest(tmp_path)
    hash_fn = build_image_hash_fn(str(manifest_path))
    hashed: list[bytes] = []
    original = segmentation_common.hash_image_bytes
    monkeypatch.setattr(segmentation_common, "hash_image_bytes", lambda data: hashed.append(data) or original(data))
    sample = PreparedSample(sample_id="sample-0", image=None, target=None)

    first = hash_fn(sample)
    second = hash_fn(sample)
    hash_fn(PreparedSample(sample_id="sample-1", image=None, target=None))

    assert first == second
    assert len(hashed) == 2  # one read per distinct sample, none for the repeat


class _CountingBatches:
    """A re-iterable batches source that records how many times it was walked."""

    def __init__(self, batches: list[TrainingBatch]) -> None:
        self.batches = batches
        self.passes = 0

    def __iter__(self) -> Iterator[TrainingBatch]:
        self.passes += 1
        yield from self.batches


def _source(values: list[int], *, batch_size: int = 2, size: tuple[int, int] = (2, 3)) -> _CountingBatches:
    batches = []
    for start in range(0, len(values), batch_size):
        chunk = values[start : start + batch_size]
        batches.append(
            TrainingBatch(
                samples=[PreparedSample(sample_id=f"s{value}", image=None, target=None) for value in chunk],
                targets=torch.stack([torch.full(size, value, dtype=torch.long) for value in chunk]),
            )
        )
    return _CountingBatches(batches)


def _replay(preloaded: PreloadedBatches) -> list[tuple[list[str], torch.Tensor]]:
    return [([sample.sample_id for sample in batch.samples], batch.targets) for batch in preloaded]


def test_preloaded_batches_walk_the_source_exactly_once() -> None:
    source = _source([0, 1, 2, 3, 4])
    preloaded = PreloadedBatches(source, batch_size=2, storage_device=torch.device("cpu"))

    for _ in range(3):
        list(preloaded)

    assert source.passes == 1


def test_preloaded_batches_replay_the_source_batches_in_order() -> None:
    source = _source([0, 1, 2, 3, 4])
    preloaded = PreloadedBatches(source, batch_size=2, storage_device=torch.device("cpu"))

    replayed = _replay(preloaded)

    assert [ids for ids, _ in replayed] == [["s0", "s1"], ["s2", "s3"], ["s4"]]
    for (_, targets), original in zip(replayed, source.batches, strict=True):
        assert torch.equal(targets.long(), original.targets)
    assert len(preloaded) == 3


def test_preloaded_batches_rebatch_to_their_own_batch_size() -> None:
    preloaded = PreloadedBatches(_source([0, 1, 2, 3, 4]), batch_size=4, storage_device=torch.device("cpu"))

    assert [ids for ids, _ in _replay(preloaded)] == [["s0", "s1", "s2", "s3"], ["s4"]]
    assert len(preloaded) == 2


@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        ([0, 14, 3], torch.uint8),
        ([0, 255], torch.uint8),
        ([-100, 2], torch.int16),  # ignore_index is negative
        ([0, 40_000], torch.int32),
        ([0, 2**40], torch.int64),
    ],
)
def test_preloaded_batches_store_targets_in_the_narrowest_exact_dtype(values: list[int], dtype: torch.dtype) -> None:
    preloaded = PreloadedBatches(_source(values), batch_size=2, storage_device=torch.device("cpu"))

    assert preloaded.targets.dtype == dtype
    assert preloaded.targets[:, 0, 0].tolist() == values


def test_preloaded_batches_keep_float_targets_unchanged() -> None:
    batch = TrainingBatch(
        samples=[PreparedSample(sample_id="a", image=None, target=None)], targets=torch.full((1, 2, 2), 0.5)
    )
    preloaded = PreloadedBatches([batch], batch_size=1, storage_device=torch.device("cpu"))

    assert preloaded.targets.dtype == torch.float32


def test_preloaded_batches_expose_their_samples_in_source_order() -> None:
    preloaded = PreloadedBatches(_source([3, 1, 2]), batch_size=2, storage_device=torch.device("cpu"))

    assert [sample.sample_id for sample in preloaded.samples] == ["s3", "s1", "s2"]


def test_preloaded_batches_without_shuffle_repeat_the_same_order_every_epoch() -> None:
    preloaded = PreloadedBatches(_source(list(range(7))), batch_size=3, storage_device=torch.device("cpu"))

    first = [ids for ids, _ in _replay(preloaded)]
    second = [ids for ids, _ in _replay(preloaded)]

    assert first == second


def test_preloaded_batches_shuffle_changes_order_between_epochs_but_keeps_pairs_aligned() -> None:
    preloaded = PreloadedBatches(
        _source(list(range(20))), batch_size=4, storage_device=torch.device("cpu"), shuffle=True, seed=3
    )

    first = _replay(preloaded)
    second = _replay(preloaded)

    first_ids = [sample_id for ids, _ in first for sample_id in ids]
    second_ids = [sample_id for ids, _ in second for sample_id in ids]
    assert sorted(first_ids) == sorted(second_ids) == sorted(f"s{value}" for value in range(20))
    assert first_ids != second_ids
    for ids, targets in first + second:
        assert [int(target[0, 0]) for target in targets] == [int(sample_id[1:]) for sample_id in ids]


def test_preloaded_batches_shuffle_order_is_a_function_of_seed_and_epoch() -> None:
    def epoch_order(*, seed: int, epoch: int) -> list[str]:
        preloaded = PreloadedBatches(
            _source(list(range(12))), batch_size=5, storage_device=torch.device("cpu"), shuffle=True, seed=seed
        )
        preloaded.set_epoch(epoch)
        return [sample_id for ids, _ in _replay(preloaded) for sample_id in ids]

    assert epoch_order(seed=1, epoch=4) == epoch_order(seed=1, epoch=4)
    assert epoch_order(seed=1, epoch=4) != epoch_order(seed=2, epoch=4)
    assert epoch_order(seed=1, epoch=4) != epoch_order(seed=1, epoch=5)


def test_preloaded_batches_set_epoch_reproduces_a_later_epoch_after_resume() -> None:
    running = PreloadedBatches(
        _source(list(range(9))), batch_size=2, storage_device=torch.device("cpu"), shuffle=True, seed=0
    )
    _replay(running)  # epoch 0
    epoch_one = [ids for ids, _ in _replay(running)]

    resumed = PreloadedBatches(
        _source(list(range(9))), batch_size=2, storage_device=torch.device("cpu"), shuffle=True, seed=0
    )
    resumed.set_epoch(1)

    assert [ids for ids, _ in _replay(resumed)] == epoch_one


def test_preloaded_batches_reject_a_negative_epoch() -> None:
    preloaded = PreloadedBatches(_source([0]), batch_size=1, storage_device=torch.device("cpu"))

    with pytest.raises(ValueError, match="epoch must be non-negative"):
        preloaded.set_epoch(-1)


def test_preloaded_batches_reject_a_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        PreloadedBatches(_source([0]), batch_size=0, storage_device=torch.device("cpu"))


def test_preloaded_batches_reject_an_empty_source() -> None:
    with pytest.raises(ValueError, match="source yielded no samples"):
        PreloadedBatches([], batch_size=2, storage_device=torch.device("cpu"))


def test_preloaded_batches_from_real_training_batches_match_every_epoch(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path, image_size=(6, 4))
    source = build_training_batches(ManifestDataset(str(manifest_path)), batch_size=3, load_images=False)
    expected = [(list(batch.samples), batch.targets) for batch in source]

    preloaded = PreloadedBatches(source, batch_size=3, storage_device=torch.device("cpu"), show_progress=True)

    for _ in range(2):
        for batch, (samples, targets) in zip(preloaded, expected, strict=True):
            assert [sample.sample_id for sample in batch.samples] == [sample.sample_id for sample in samples]
            assert torch.equal(batch.targets.long(), targets)


_needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA-enabled machine")


@_needs_cuda
def test_preloaded_batches_pin_host_targets_for_async_copies() -> None:
    preloaded = PreloadedBatches(_source([0, 1, 2]), batch_size=2, storage_device=torch.device("cpu"), pin_memory=True)

    batch = next(iter(preloaded))

    assert batch.targets.is_pinned()
    assert batch.targets.tolist() == [[[0] * 3] * 2, [[1] * 3] * 2]


@_needs_cuda
def test_preloaded_batches_on_a_cuda_device_yield_device_targets() -> None:
    preloaded = PreloadedBatches(
        _source([0, 1, 2]), batch_size=2, storage_device=torch.device("cuda"), shuffle=True, seed=0
    )

    batches = list(preloaded)

    assert all(batch.targets.device.type == "cuda" for batch in batches)
    for batch in batches:
        assert [int(target[0, 0]) for target in batch.targets] == [int(s.sample_id[1:]) for s in batch.samples]
