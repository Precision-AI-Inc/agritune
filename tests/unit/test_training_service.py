# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.services.training_service internals not already covered by
the end-to-end integration test."""

import pytest

from precisionai.agritune.services.training_service import _build_decoder
from precisionai.agritune.tasks.segmentation.decoders.linear import LinearProbeDecoder
from precisionai.agritune.tasks.segmentation.decoders.token_fpn import TokenFPNDecoder


def test_build_decoder_linear() -> None:
    decoder = _build_decoder("linear", patch_dim=8, cls_dim=None, num_classes=2, output_size=(4, 4))
    assert isinstance(decoder, LinearProbeDecoder)


def test_build_decoder_token_fpn() -> None:
    decoder = _build_decoder("token_fpn", patch_dim=8, cls_dim=None, num_classes=2, output_size=(4, 4))
    assert isinstance(decoder, TokenFPNDecoder)


def test_build_decoder_unsupported_name_raises() -> None:
    with pytest.raises(ValueError, match="unsupported decoder"):
        _build_decoder("unknown", patch_dim=8, cls_dim=None, num_classes=2, output_size=(4, 4))
