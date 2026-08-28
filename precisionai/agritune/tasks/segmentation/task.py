# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""``SegmentationTask`` — binds a decoder to a loss, satisfying the ``Task`` protocol."""

import torch
from torch import nn

from precisionai.agritune.schemas.features import EncoderFeatures
from precisionai.agritune.tasks.segmentation.losses import SegmentationLoss


class SegmentationTask:
    """The first downstream task: semantic segmentation on frozen encoder features.

    Parameters
    ----------
    decoder : torch.nn.Module
        Any of the decoders in :mod:`precisionai.agritune.tasks.segmentation.decoders`
        (:class:`~precisionai.agritune.tasks.segmentation.decoders.mlp_probe.MLPProbeDecoder`,
        :class:`~precisionai.agritune.tasks.segmentation.decoders.token_fpn.TokenFPNDecoder`,
        etc.), or any module satisfying :class:`~precisionai.agritune.schemas.protocols.Decoder`.
    loss : SegmentationLoss
        The configured loss to train against.
    """

    def __init__(self, decoder: nn.Module, loss: SegmentationLoss) -> None:
        self.decoder = decoder
        self.loss = loss

    def forward(self, features: EncoderFeatures) -> torch.Tensor:
        """Run the decoder, returning per-pixel class logits."""
        return self.decoder(features)

    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the configured segmentation loss for one batch."""
        return self.loss(outputs, targets)
