# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""TokenFPN — a stronger segmentation decoder than the linear probe.

patch tokens -> channel projection -> reshape to spatial grid -> pseudo feature pyramid ->
convolutional refinement -> upsampling -> segmentation logits.

"Pseudo" because a plain ViT encoder exposes only one layer of patch tokens, not a true
multi-scale pyramid — refinement happens at the single patch-grid resolution before upsampling.
CLS fusion is optional and never required (many encoders/models produce no CLS token at all).
"""

from enum import Enum

import torch
from torch import nn
from torch.nn import functional

from precisionai.agritune.schemas.features import EncoderFeatures


class CLSFusion(str, Enum):
    """How (if at all) the CLS token is fused into the spatial feature map."""

    NONE = "none"
    CONCAT = "concat"
    FILM = "film"


class TokenFPNDecoder(nn.Module):
    """Channel-projected, convolutionally refined patch tokens, with optional CLS fusion.

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
    cls_dim : int | None, optional
        CLS embedding dimension; required when ``cls_fusion`` is not ``CLSFusion.NONE``.
    hidden_dim : int, optional
        Channel width used throughout the refinement stack.
    num_layers : int, optional
        Number of stacked 3x3 conv + ReLU blocks in the refinement stack.
    cls_fusion : CLSFusion, optional
        How to fuse the CLS token: ``NONE`` (ignore it), ``CONCAT`` (project and concatenate as
        extra spatial channels), or ``FILM`` (feature-wise linear modulation: scale and shift the
        spatial feature map).

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
        cls_dim: int | None = None,
        hidden_dim: int = 128,
        num_layers: int = 2,
        cls_fusion: CLSFusion = CLSFusion.NONE,
    ) -> None:
        super().__init__()
        if cls_fusion is not CLSFusion.NONE and cls_dim is None:
            raise ValueError(f"cls_fusion={cls_fusion.value!r} requires cls_dim to be set")
        if num_layers < 1:
            raise ValueError(f"num_layers must be positive; got {num_layers}")

        self.output_size = output_size
        self.cls_fusion = cls_fusion
        self.channel_proj = nn.Conv2d(patch_dim, hidden_dim, kernel_size=1)

        refine_in_channels = hidden_dim
        self.cls_proj: nn.Module | None = None
        if cls_fusion is CLSFusion.CONCAT:
            if cls_dim is None:  # pragma: no cover — excluded by the check above
                raise RuntimeError("cls_dim must be set for cls_fusion=concat")
            self.cls_proj = nn.Linear(cls_dim, hidden_dim)
            refine_in_channels = hidden_dim * 2
        elif cls_fusion is CLSFusion.FILM:
            if cls_dim is None:  # pragma: no cover — excluded by the check above
                raise RuntimeError("cls_dim must be set for cls_fusion=film")
            self.cls_proj = nn.Linear(cls_dim, hidden_dim * 2)

        refine_layers: list[nn.Module] = []
        in_channels = refine_in_channels
        for _ in range(num_layers):
            refine_layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1))
            refine_layers.append(nn.ReLU(inplace=True))
            in_channels = hidden_dim
        self.refine = nn.Sequential(*refine_layers)
        self.classifier = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, features: EncoderFeatures) -> torch.Tensor:
        """Return per-pixel class logits, shape ``(B, num_classes, *output_size)``."""
        height, width = features.uniform_patch_grid()
        batch_size, _, patch_dim = features.patch_tokens.shape

        grid = features.patch_tokens.reshape(batch_size, height, width, patch_dim).permute(0, 3, 1, 2)
        grid = self.channel_proj(grid)  # (B, hidden_dim, H, W)
        grid = self._fuse_cls(grid, features)

        refined = self.refine(grid)
        logits = self.classifier(refined)
        return functional.interpolate(logits, size=self.output_size, mode="bilinear", align_corners=False)

    def _fuse_cls(self, grid: torch.Tensor, features: EncoderFeatures) -> torch.Tensor:
        if self.cls_fusion is CLSFusion.NONE:
            return grid
        if features.cls_tokens is None:
            raise ValueError(f"cls_fusion={self.cls_fusion.value!r} requires cls_tokens in EncoderFeatures")

        if self.cls_proj is None:  # pragma: no cover — excluded by the __init__ validation
            raise RuntimeError("cls_proj was not initialized despite cls_fusion != NONE")
        _, _, height, width = grid.shape

        if self.cls_fusion is CLSFusion.CONCAT:
            cls = self.cls_proj(features.cls_tokens)  # (B, hidden_dim)
            cls = cls[:, :, None, None].expand(-1, -1, height, width)
            return torch.cat([grid, cls], dim=1)

        # FILM: split into per-sample scale and shift, broadcast spatially.
        scale, shift = self.cls_proj(features.cls_tokens).chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        return grid * (1.0 + scale) + shift
