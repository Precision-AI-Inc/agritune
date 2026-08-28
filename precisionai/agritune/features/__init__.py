# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0
"""FeatureProvider, FeatureStore, cache keys, and resumable precomputation."""

from precisionai.agritune.features.errors import FeatureNotCachedError
from precisionai.agritune.features.keys import (
    FEATURE_SCHEMA_VERSION,
    EncoderFingerprint,
    compute_feature_key,
    hash_augmentation,
    hash_image_bytes,
)
from precisionai.agritune.features.manifest import FeatureManifest
from precisionai.agritune.features.precompute import PrecomputeStats, precompute_features
from precisionai.agritune.features.provider import CachedFeatureProvider
from precisionai.agritune.features.store import DirectoryFeatureStore, FeatureSummary, ShardedFeatureStore

__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "CachedFeatureProvider",
    "DirectoryFeatureStore",
    "EncoderFingerprint",
    "FeatureManifest",
    "FeatureNotCachedError",
    "FeatureSummary",
    "PrecomputeStats",
    "ShardedFeatureStore",
    "compute_feature_key",
    "hash_augmentation",
    "hash_image_bytes",
    "precompute_features",
]
