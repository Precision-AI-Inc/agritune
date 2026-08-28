# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.services.segmentation_common."""

from pathlib import Path

import pytest
import torch

from precisionai.agritune.data.dataset import ManifestDataset
from precisionai.agritune.encoder.fake import FakeEncoderBackend
from precisionai.agritune.features.keys import EncoderFingerprint
from precisionai.agritune.features.store import DirectoryFeatureStore
from precisionai.agritune.schemas.samples import PreparedSample
from precisionai.agritune.services.feature_service import build_features
from precisionai.agritune.services.segmentation_common import (
    build_cached_feature_provider,
    build_decoder,
    build_prediction_batches,
    build_training_batches,
    mask_to_target_tensor,
    probe_feature_dims,
)
from precisionai.agritune.tasks.segmentation.decoders.linear import LinearProbeDecoder
from precisionai.agritune.tasks.segmentation.decoders.token_fpn import TokenFPNDecoder
from tests.fixtures.manifest_factory import build_manifest

_FINGERPRINT = EncoderFingerprint(model="fake-encoder", revision="fake-v1", preprocessing="resize=8x8")


def test_build_decoder_linear() -> None:
    decoder = build_decoder("linear", patch_dim=8, cls_dim=None, num_classes=2, output_size=(4, 4))
    assert isinstance(decoder, LinearProbeDecoder)


def test_build_decoder_token_fpn() -> None:
    decoder = build_decoder("token_fpn", patch_dim=8, cls_dim=None, num_classes=2, output_size=(4, 4))
    assert isinstance(decoder, TokenFPNDecoder)


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


def test_build_prediction_batches_have_no_targets(tmp_path: Path) -> None:
    manifest_path = build_manifest(tmp_path)
    dataset = ManifestDataset(manifest_path)
    batches = build_prediction_batches(dataset, batch_size=2)
    assert [len(batch) for batch in batches] == [2, 2]
    assert all(sample.target is None for batch in batches for sample in batch)


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
