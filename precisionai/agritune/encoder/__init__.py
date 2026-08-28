# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0
"""EncoderBackend protocol, FakeEncoderBackend/RemoteEncoderBackend, and the EncoderGateway.

Only ``precisionai.agritune.features`` may import from this package — see CLAUDE.md's
architectural rule: the trainer and decoders must never call the encoder API directly.
"""

from precisionai.agritune.encoder.base import EncoderBackend, ImageInput
from precisionai.agritune.encoder.batching import batch_items
from precisionai.agritune.encoder.errors import (
    EncoderError,
    EncoderRateLimitError,
    EncoderServerError,
    EncoderTimeoutError,
    MalformedEncoderResponseError,
)
from precisionai.agritune.encoder.fake import FakeEncoderBackend, FakeEncoderConfig
from precisionai.agritune.encoder.gateway import EncoderGateway, GatewayConfig, GatewayMetrics
from precisionai.agritune.encoder.rate_limiter import RateLimiter, RateLimiterConfig
from precisionai.agritune.encoder.remote import RemoteEncoderBackend, RemoteEncoderConfig
from precisionai.agritune.encoder.retry import RetryExhaustedError, RetryPolicy, RetryPolicyConfig
from precisionai.agritune.encoder.validation import (
    EncoderConsistencyError,
    EncoderDimensionFingerprint,
    EncoderResponseValidator,
)

__all__ = [
    "EncoderBackend",
    "EncoderConsistencyError",
    "EncoderDimensionFingerprint",
    "EncoderError",
    "EncoderGateway",
    "EncoderRateLimitError",
    "EncoderResponseValidator",
    "EncoderServerError",
    "EncoderTimeoutError",
    "FakeEncoderBackend",
    "FakeEncoderConfig",
    "GatewayConfig",
    "GatewayMetrics",
    "ImageInput",
    "MalformedEncoderResponseError",
    "RateLimiter",
    "RateLimiterConfig",
    "RemoteEncoderBackend",
    "RemoteEncoderConfig",
    "RetryExhaustedError",
    "RetryPolicy",
    "RetryPolicyConfig",
    "batch_items",
]
