# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Re-exports the encoder-facing contract defined in ``precisionai.agritune.schemas``.

The :class:`~precisionai.agritune.schemas.protocols.EncoderBackend` protocol itself lives in
``schemas.protocols`` (Phase 1) so it has no dependency on the concrete encoder package; this
module exists so encoder implementations can import it from a natural, package-local location.
"""

from precisionai.agritune.schemas.protocols import EncoderBackend, ImageInput

__all__ = [
    "EncoderBackend",
    "ImageInput",
]
