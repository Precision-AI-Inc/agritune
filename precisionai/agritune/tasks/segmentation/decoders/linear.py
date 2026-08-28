# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""The linear probe segmentation decoder — the diagnostic baseline.

patch tokens -> linear projection -> per-patch class logits -> reshape via patch_grid -> upsample.

If a more complex decoder (e.g. TokenFPN) does not beat this, something upstream is wrong — see
``agritune_implementation_plan.md`` §11.
"""

import torch
from torch import nn
from torch.nn import functional

from precisionai.agritune.schemas.features import EncoderFeatures


class LinearProbeDecoder(nn.Module):
    """A single linear layer mapping patch tokens to per-pixel class logits.

    Requires every sample in a batch to share the same patch grid (the normal case when a fixed
    input resolution is enforced by the augmentation pipeline) — see
    :meth:`EncoderFeatures.uniform_patch_grid`.

    Parameters
    ----------
    patch_dim : int
        Patch embedding dimension ``D``.
    num_classes : int
        Number of segmentation classes.
    output_size : tuple[int, int]
        ``(height, width)`` to upsample logits to.
    """

    def __init__(self, *, patch_dim: int, num_classes: int, output_size: tuple[int, int]) -> None:
        super().__init__()
        self.output_size = output_size
        self.projection = nn.Linear(patch_dim, num_classes)

    def forward(self, features: EncoderFeatures) -> torch.Tensor:
        """Return per-pixel class logits, shape ``(B, num_classes, *output_size)``."""
        height, width = features.uniform_patch_grid()
        logits = self.projection(features.patch_tokens)  # (B, N, num_classes)
        logits = logits.reshape(features.batch_size, height, width, -1).permute(0, 3, 1, 2)
        return functional.interpolate(logits, size=self.output_size, mode="bilinear", align_corners=False)
