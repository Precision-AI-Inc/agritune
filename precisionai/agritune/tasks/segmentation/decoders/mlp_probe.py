# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""The MLP probe segmentation decoder — a depth-configurable generalization of the linear probe.

patch tokens -> stack of per-patch linear (+ ReLU) layers -> per-patch class logits -> reshape via
patch_grid -> upsample. With ``hidden_dims=()`` (the default) this is architecturally identical to
the former ``LinearProbeDecoder`` — a single linear projection and nothing else — so it still
serves as the diagnostic baseline: if a more complex decoder does not beat it, something upstream
is wrong.
"""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional

from precisionai.agritune.schemas.features import EncoderFeatures


class MLPProbeDecoder(nn.Module):
    """A per-patch MLP mapping patch tokens to per-pixel class logits.

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
    hidden_dims : Sequence[int], optional
        Widths of intermediate layers. Empty (the default) collapses to a single
        ``nn.Linear(patch_dim, num_classes)`` — the original linear-probe behavior.
    dropout : float, optional
        Dropout probability applied after each hidden layer's activation; ``0.0`` disables it.
        Never applied after the final classification layer.
    """

    def __init__(
        self,
        *,
        patch_dim: int,
        num_classes: int,
        output_size: tuple[int, int],
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.hidden_dims = tuple(hidden_dims)

        layers: list[nn.Module] = []
        in_dim = patch_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, features: EncoderFeatures) -> torch.Tensor:
        """Return per-pixel class logits, shape ``(B, num_classes, *output_size)``."""
        height, width = features.uniform_patch_grid()
        logits = self.mlp(features.patch_tokens)  # (B, N, num_classes)
        logits = logits.reshape(features.batch_size, height, width, -1).permute(0, 3, 1, 2)
        return functional.interpolate(logits, size=self.output_size, mode="bilinear", align_corners=False)
