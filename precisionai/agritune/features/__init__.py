# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0
"""FeatureProvider, FeatureStore, cache keys, and resumable precomputation."""

from precisionai.agritune.features.errors import FeatureNotCachedError
from precisionai.agritune.features.integrity import VerificationReport, verify_store
from precisionai.agritune.features.keys import (
    FEATURE_SCHEMA_VERSION,
    EncoderFingerprint,
    compute_feature_key,
    hash_augmentation,
    hash_image_bytes,
)
from precisionai.agritune.features.manifest import FeatureManifest
from precisionai.agritune.features.migrate import MigrationReport, migrate_store
from precisionai.agritune.features.precompute import PrecomputeStats, precompute_features
from precisionai.agritune.features.prefetch import PrefetchingFeatureProvider
from precisionai.agritune.features.provider import CachedFeatureProvider, HybridFeatureProvider, OnlineFeatureProvider
from precisionai.agritune.features.store import (
    FEATURE_STORE_TYPES,
    DirectoryFeatureStore,
    FeatureSummary,
    ShardedFeatureStore,
    build_feature_store,
)

__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "FEATURE_STORE_TYPES",
    "CachedFeatureProvider",
    "DirectoryFeatureStore",
    "EncoderFingerprint",
    "FeatureManifest",
    "FeatureNotCachedError",
    "FeatureSummary",
    "HybridFeatureProvider",
    "MigrationReport",
    "OnlineFeatureProvider",
    "PrecomputeStats",
    "PrefetchingFeatureProvider",
    "ShardedFeatureStore",
    "VerificationReport",
    "build_feature_store",
    "compute_feature_key",
    "hash_augmentation",
    "hash_image_bytes",
    "migrate_store",
    "precompute_features",
    "verify_store",
]
