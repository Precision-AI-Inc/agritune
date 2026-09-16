# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""PPM — the PSPNet-style pyramid pooling segmentation decoder.

patch tokens -> spatial grid -> parallel adaptive average pools at several bin sizes, each
channel-reduced and upsampled back to the patch grid -> concatenation with the original grid ->
convolutional fusion -> per-pixel class logits -> upsample. See Zhao et al., "Pyramid Scene Parsing
Network" (PSPNet, CVPR 2017); adapted here to the single patch-grid resolution the frozen ViT
features expose.
"""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional

from precisionai.agritune.schemas.features import EncoderFeatures


class PyramidPoolingDecoder(nn.Module):
    """Multi-bin adaptive-pooling context segmentation decoder.

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
        Channel width of the fused feature map (before the final classifier).
    pool_sizes : Sequence[int], optional
        Adaptive-average-pool output sizes (each pools the grid to a ``size x size`` bin map).
    num_layers : int, optional
        Number of stacked 3x3 conv + ReLU blocks fusing the pooled features together.

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
        pool_sizes: Sequence[int] = (1, 2, 3, 6),
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be positive; got {num_layers}")
        self.output_size = output_size
        self.pool_sizes = tuple(pool_sizes)
        pool_channels = max(hidden_dim // max(len(self.pool_sizes), 1), 1)

        self.pools = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(size),
                    nn.Conv2d(patch_dim, pool_channels, kernel_size=1),
                    nn.ReLU(inplace=True),
                )
                for size in self.pool_sizes
            ]
        )

        fused_channels = patch_dim + pool_channels * len(self.pool_sizes)
        fuse_layers: list[nn.Module] = []
        in_channels = fused_channels
        for _ in range(num_layers):
            fuse_layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1))
            fuse_layers.append(nn.ReLU(inplace=True))
            in_channels = hidden_dim
        self.fuse = nn.Sequential(*fuse_layers)
        self.classifier = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, features: EncoderFeatures) -> torch.Tensor:
        """Return per-pixel class logits, shape ``(B, num_classes, *output_size)``."""
        height, width = features.uniform_patch_grid()
        batch_size, _, patch_dim = features.patch_tokens.shape
        grid = features.patch_tokens.reshape(batch_size, height, width, patch_dim).permute(0, 3, 1, 2)

        pooled_features = [grid]
        for pool in self.pools:
            pooled = pool(grid)
            pooled = functional.interpolate(pooled, size=(height, width), mode="bilinear", align_corners=False)
            pooled_features.append(pooled)

        fused = self.fuse(torch.cat(pooled_features, dim=1))
        logits = self.classifier(fused)
        return functional.interpolate(logits, size=self.output_size, mode="bilinear", align_corners=False)
