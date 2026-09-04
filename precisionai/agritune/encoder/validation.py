# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Response validation: flags a silent encoder/model swap mid-run.

The hosted embeddings API exposes no queryable encoder revision (see ``docs/encoder.md``) — the
server can change the model behind an alias without any client-visible version bump.
:class:`EncoderResponseValidator` is the safety net: it fingerprints the dimensions observed on
the first response of a run and raises if a later response's dimensions drift, since two encoder
versions producing features of the same shape is the one case this cannot detect.
"""

from dataclasses import dataclass

import torch

from precisionai.agritune.encoder.errors import EncoderError
from precisionai.agritune.logging import get_logger
from precisionai.agritune.schemas.features import EncoderFeatures

logger = get_logger(__name__)


@dataclass(frozen=True)
class EncoderDimensionFingerprint:
    """The observable dimensions of one :class:`EncoderFeatures` batch.

    Attributes
    ----------
    patch_dim : int
        Patch embedding dimension.
    cls_dim : int | None
        CLS embedding dimension, or ``None`` if no CLS token is produced.
    patch_dtype : torch.dtype
        Patch-token dtype observed from the encoder response.
    cls_dtype : torch.dtype | None
        CLS-token dtype, or ``None`` if no CLS token is produced.
    """

    patch_dim: int
    cls_dim: int | None
    patch_dtype: torch.dtype
    cls_dtype: torch.dtype | None


class EncoderConsistencyError(EncoderError):
    """Raised when observed encoder dimensions change mid-run.

    This most likely means the server swapped the model backing the configured alias.
    """


class EncoderResponseValidator:
    """Tracks the first-seen :class:`EncoderDimensionFingerprint` for a run and flags drift."""

    def __init__(self) -> None:
        self._fingerprint: EncoderDimensionFingerprint | None = None

    def validate(self, features: EncoderFeatures) -> None:
        """Check ``features`` against the fingerprint established by the first call.

        Parameters
        ----------
        features : EncoderFeatures
            The batch just received from the encoder.

        Raises
        ------
        EncoderConsistencyError
            If this is not the first call and the observed dimensions differ from the first.
        """
        patch_dim = features.patch_tokens.shape[-1]
        cls_dim = features.cls_tokens.shape[-1] if features.cls_tokens is not None else None
        cls_dtype = features.cls_tokens.dtype if features.cls_tokens is not None else None
        fingerprint = EncoderDimensionFingerprint(
            patch_dim=patch_dim,
            cls_dim=cls_dim,
            patch_dtype=features.patch_tokens.dtype,
            cls_dtype=cls_dtype,
        )

        if self._fingerprint is None:
            self._fingerprint = fingerprint
            return

        if fingerprint != self._fingerprint:
            logger.warning(
                "encoder dimensions changed mid-run: first saw %s, now saw %s", self._fingerprint, fingerprint
            )
            raise EncoderConsistencyError(
                f"encoder dimensions changed mid-run: first saw {self._fingerprint}, now saw "
                f"{fingerprint} — the server may have swapped the model behind the configured alias"
            )
