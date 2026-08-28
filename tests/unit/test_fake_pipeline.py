# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Phase 1 definition of done: FakeDataset -> FakeFeatureProvider -> FakeDecoder -> FakeTask runs
end-to-end without any encoder API call, and each fake satisfies its corresponding protocol."""

from precisionai.agritune.schemas.protocols import Decoder, FeatureProvider, Task
from precisionai.agritune.schemas.samples import PreparedSample
from tests.fixtures.fake_pipeline import FakeDataset, FakeDecoder, FakeFeatureProvider, FakeTask


def test_fake_pipeline_end_to_end() -> None:
    dataset = FakeDataset(num_samples=4)
    provider = FakeFeatureProvider(patch_dim=8, cls_dim=5, patch_grid=(2, 3))
    decoder = FakeDecoder(patch_dim=8, num_classes=3)
    task = FakeTask(decoder)

    prepared = [PreparedSample(sample_id=s.sample_id, image=s.image, target=s.target) for s in dataset]
    features = provider.get_features(prepared)
    outputs = task.forward(features)
    loss = task.compute_loss(outputs, targets=None)

    assert outputs.shape == (4, 6, 3)
    assert loss.item() == 0.0


def test_fake_feature_provider_satisfies_protocol() -> None:
    assert isinstance(FakeFeatureProvider(), FeatureProvider)


def test_fake_decoder_satisfies_protocol() -> None:
    assert isinstance(FakeDecoder(patch_dim=8, num_classes=3), Decoder)


def test_fake_task_satisfies_protocol() -> None:
    assert isinstance(FakeTask(FakeDecoder(patch_dim=8, num_classes=3)), Task)
