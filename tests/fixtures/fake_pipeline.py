# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Minimal fake implementations proving the Phase 1 schema/protocol wiring end-to-end.

``FakeDataset -> FakeFeatureProvider -> FakeDecoder -> FakeTask`` runs with no encoder API call
and no I/O. This is distinct from ``FakeEncoderBackend`` (Phase 4, ``precisionai.agritune.encoder``),
which the rest of the test suite uses to exercise the real encoder/gateway/feature-store stack.
"""

from collections.abc import Sequence
from typing import Any

import torch

from precisionai.agritune.schemas.features import EncoderFeatures
from precisionai.agritune.schemas.samples import PreparedSample, Sample


class FakeDataset:
    """An in-memory sequence of :class:`Sample` objects — no file I/O."""

    def __init__(self, num_samples: int = 4) -> None:
        self._samples = [
            Sample(sample_id=f"sample-{i}", image=f"image-{i}", target=f"mask-{i}") for i in range(num_samples)
        ]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Sample:
        return self._samples[index]


class FakeFeatureProvider:
    """Produces deterministic :class:`EncoderFeatures` directly — never calls an encoder."""

    def __init__(
        self,
        *,
        patch_dim: int = 8,
        cls_dim: int | None = 5,
        patch_grid: tuple[int, int] = (2, 3),
    ) -> None:
        self._patch_dim = patch_dim
        self._cls_dim = cls_dim
        self._patch_grid = patch_grid

    def get_features(self, samples: Sequence[PreparedSample]) -> EncoderFeatures:
        """Return deterministic features for a batch of prepared samples."""
        batch_size = len(samples)
        height, width = self._patch_grid
        num_patches = height * width
        generator = torch.Generator().manual_seed(0)
        return EncoderFeatures(
            patch_tokens=torch.randn(batch_size, num_patches, self._patch_dim, generator=generator),
            cls_tokens=(
                torch.randn(batch_size, self._cls_dim, generator=generator) if self._cls_dim is not None else None
            ),
            patch_grid=torch.tensor([[height, width]] * batch_size, dtype=torch.long),
            valid_patch_mask=None,
            image_sizes=[(224, 224)] * batch_size,
            encoder_model="fake",
            encoder_revision=None,
        )


class FakeDecoder:
    """A zero-initialized linear projection of patch tokens to per-patch class logits."""

    def __init__(self, *, patch_dim: int, num_classes: int) -> None:
        self._weight = torch.zeros(num_classes, patch_dim)

    def forward(self, features: EncoderFeatures) -> torch.Tensor:
        """Project patch tokens to per-patch class logits, shape ``(B, N, num_classes)``."""
        return features.patch_tokens @ self._weight.T


class FakeTask:
    """Binds a :class:`FakeDecoder` to a trivial loss, for pipeline wiring tests only."""

    def __init__(self, decoder: FakeDecoder) -> None:
        self._decoder = decoder

    def forward(self, features: EncoderFeatures) -> torch.Tensor:
        """Run the wrapped decoder."""
        return self._decoder.forward(features)

    def compute_loss(self, outputs: torch.Tensor, targets: Any) -> torch.Tensor:
        """Return a trivial scalar loss — this fake exists to test wiring, not learning."""
        del targets
        return outputs.pow(2).mean()
