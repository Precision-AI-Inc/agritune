# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Segmenter-style mask transformer segmentation decoder.

A set of learnable per-class embeddings is concatenated with the (projected) patch tokens and
jointly processed by shared self-attention transformer layers; splitting the joint output back
apart and taking a scaled dot product between the patch and class embeddings yields per-pixel,
per-class mask logits directly — no convolutional decoder head is needed. See Strudel et al.,
"Segmenter: Transformer for Semantic Segmentation" (ICCV 2021).
"""

import torch
from torch import nn
from torch.nn import functional

from precisionai.agritune.schemas.features import EncoderFeatures


class SegmenterMaskTransformerDecoder(nn.Module):
    """Joint patch/class-token transformer segmentation decoder.

    Requires every sample in a batch to share the same patch grid — see
    :meth:`EncoderFeatures.uniform_patch_grid`.

    Parameters
    ----------
    patch_dim : int
        Patch embedding dimension.
    num_classes : int
        Number of segmentation classes — one learnable class embedding is created per class.
    output_size : tuple[int, int]
        ``(height, width)`` to upsample logits to.
    hidden_dim : int, optional
        Transformer model width. Must be divisible by ``num_heads``.
    num_layers : int, optional
        Number of shared self-attention transformer encoder layers.
    num_heads : int, optional
        Attention heads per layer.
    mlp_ratio : int, optional
        Feedforward expansion ratio inside each transformer layer.
    dropout : float, optional
        Dropout probability inside the transformer layers.

    Raises
    ------
    ValueError
        If ``hidden_dim`` is not divisible by ``num_heads``.
    """

    def __init__(
        self,
        *,
        patch_dim: int,
        num_classes: int,
        output_size: tuple[int, int],
        hidden_dim: int = 192,
        num_layers: int = 2,
        num_heads: int = 3,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")

        self.output_size = output_size
        self.num_classes = num_classes
        self.scale = hidden_dim**-0.5

        self.patch_proj = nn.Linear(patch_dim, hidden_dim)
        self.class_embeddings = nn.Parameter(torch.randn(num_classes, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * mlp_ratio,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.patch_norm = nn.LayerNorm(hidden_dim)
        self.class_norm = nn.LayerNorm(hidden_dim)

    def forward(self, features: EncoderFeatures) -> torch.Tensor:
        """Return per-pixel class logits, shape ``(B, num_classes, *output_size)``."""
        height, width = features.uniform_patch_grid()
        batch_size = features.batch_size

        patches = self.patch_proj(features.patch_tokens)  # (B, N, hidden_dim)
        classes = self.class_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # (B, num_classes, hidden_dim)

        joint = self.transformer(torch.cat([patches, classes], dim=1))
        num_patches = patches.shape[1]
        patches_out = self.patch_norm(joint[:, :num_patches])
        classes_out = self.class_norm(joint[:, num_patches:])

        masks = torch.einsum("bnd,bcd->bnc", patches_out, classes_out) * self.scale
        logits = masks.reshape(batch_size, height, width, self.num_classes).permute(0, 3, 1, 2)
        return functional.interpolate(logits, size=self.output_size, mode="bilinear", align_corners=False)
