# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""MaskFormer/Mask2Former-style mask-classification segmentation decoder.

A fixed set of learnable object queries attends to the patch tokens through transformer decoder
layers (cross-attention + self-attention); each query is projected to both a class distribution
(over ``num_classes`` classes plus a "no object" class) and a mask embedding. Mask embeddings are
dot-producted against a convolutional pixel embedding of the patch grid to produce one soft mask
per query; masks are combined, weighted by each query's class probabilities, into dense per-pixel
class scores — the same combination MaskFormer/Mask2Former use to derive a semantic segmentation
map from their query outputs at inference time. See Cheng et al., "Per-Pixel Classification is Not
All You Need for Semantic Segmentation" (MaskFormer, NeurIPS 2021) and "Masked-attention Mask
Transformer for Universal Image Segmentation" (Mask2Former, CVPR 2022).

A single-scale simplification: this decoder decouples the number of queries from the number of
classes as the original papers do, but — unlike them — trains against the same dense per-pixel
loss as every other decoder in this package rather than a bipartite-matching set loss, since a
single dense per-pixel loss is the contract every
:class:`~precisionai.agritune.schemas.protocols.Decoder` in this repository is built against.
"""

import torch
from torch import nn
from torch.nn import functional

from precisionai.agritune.schemas.features import EncoderFeatures


class MaskFormerDecoder(nn.Module):
    """Query-based mask-classification segmentation decoder.

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
        Transformer/pixel-embedding width. Must be divisible by ``num_heads``.
    num_queries : int, optional
        Number of learnable object queries — independent of ``num_classes``.
    num_layers : int, optional
        Number of transformer decoder layers the queries pass through.
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
        hidden_dim: int = 128,
        num_queries: int = 32,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")

        self.output_size = output_size
        self.num_classes = num_classes

        self.pixel_proj = nn.Conv2d(patch_dim, hidden_dim, kernel_size=1)
        self.patch_proj = nn.Linear(patch_dim, hidden_dim)
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * mlp_ratio,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.class_head = nn.Linear(hidden_dim, num_classes + 1)  # +1 for the "no object" class
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, features: EncoderFeatures) -> torch.Tensor:
        """Return per-pixel class scores, shape ``(B, num_classes, *output_size)``."""
        height, width = features.uniform_patch_grid()
        batch_size, _, patch_dim = features.patch_tokens.shape
        grid = features.patch_tokens.reshape(batch_size, height, width, patch_dim).permute(0, 3, 1, 2)
        pixel_embed = self.pixel_proj(grid)  # (B, hidden_dim, H, W)

        memory = self.patch_proj(features.patch_tokens)  # (B, N, hidden_dim)
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        decoded = self.transformer(queries, memory)  # (B, num_queries, hidden_dim)

        class_probs = functional.softmax(self.class_head(decoded), dim=-1)[..., : self.num_classes]
        mask_probs = torch.sigmoid(torch.einsum("bqd,bdhw->bqhw", self.mask_embed(decoded), pixel_embed))

        combined = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
        return functional.interpolate(combined, size=self.output_size, mode="bilinear", align_corners=False)
