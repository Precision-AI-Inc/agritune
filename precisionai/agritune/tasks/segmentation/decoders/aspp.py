# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""ASPP — Atrous Spatial Pyramid Pooling segmentation decoder.

patch tokens -> spatial grid -> parallel dilated convolutions at several rates plus a
global-context branch -> concatenation -> projection -> per-pixel class logits -> upsample.
Captures multi-scale context from the single patch-grid resolution the frozen ViT features expose
(``EncoderFeatures`` carries only one layer of tokens), without needing multiple encoder layers —
see Chen et al., "Rethinking Atrous Convolution for Semantic Image Segmentation" (DeepLabv3).
"""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional

from precisionai.agritune.schemas.features import EncoderFeatures


class ASPPDecoder(nn.Module):
    """Dilated-convolution multi-scale-context segmentation decoder.

    Requires every sample in a batch to share the same patch grid — see
    :meth:`EncoderFeatures.uniform_patch_grid`.

    Parameters
    ----------
    patch_dim : int
        Patch embedding dimension.
    num_classes : int
        Number of segmentation classes.
    output_size : tuple[int, int]
        ``(height, width)`` to upsample logits to.
    hidden_dim : int, optional
        Channel width of every branch and the fused projection.
    atrous_rates : Sequence[int], optional
        Dilation rates for the parallel 3x3 branches, in addition to a 1x1 branch and a
        global-average-pooling branch.
    num_layers : int, optional
        Number of stacked 1x1 conv + ReLU blocks in the post-fusion projection.

    Raises
    ------
    ValueError
        If ``num_layers`` is not positive.
    """

    def __init__(
        self,
        *,
        patch_dim: int,
        num_classes: int,
        output_size: tuple[int, int],
        hidden_dim: int = 128,
        atrous_rates: Sequence[int] = (6, 12, 18),
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be positive; got {num_layers}")
        self.output_size = output_size
        self.atrous_rates = tuple(atrous_rates)

        self.branches = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(patch_dim, hidden_dim, kernel_size=1), nn.ReLU(inplace=True))]
        )
        for rate in self.atrous_rates:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(patch_dim, hidden_dim, kernel_size=3, padding=rate, dilation=rate),
                    nn.ReLU(inplace=True),
                )
            )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_branch = nn.Sequential(nn.Conv2d(patch_dim, hidden_dim, kernel_size=1), nn.ReLU(inplace=True))

        num_branches = len(self.branches) + 1
        project_layers: list[nn.Module] = []
        in_channels = hidden_dim * num_branches
        for _ in range(num_layers):
            project_layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
            project_layers.append(nn.ReLU(inplace=True))
            in_channels = hidden_dim
        self.project = nn.Sequential(*project_layers)
        self.classifier = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, features: EncoderFeatures) -> torch.Tensor:
        """Return per-pixel class logits, shape ``(B, num_classes, *output_size)``."""
        height, width = features.uniform_patch_grid()
        batch_size, _, patch_dim = features.patch_tokens.shape
        grid = features.patch_tokens.reshape(batch_size, height, width, patch_dim).permute(0, 3, 1, 2)

        outputs = [branch(grid) for branch in self.branches]
        pooled = self.global_branch(self.global_pool(grid))
        pooled = functional.interpolate(pooled, size=(height, width), mode="bilinear", align_corners=False)
        outputs.append(pooled)

        fused = self.project(torch.cat(outputs, dim=1))
        logits = self.classifier(fused)
        return functional.interpolate(logits, size=self.output_size, mode="bilinear", align_corners=False)
