# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Routes for ``agritune features build/verify/inspect/clean`` — see ``precisionai.agritune.cli.handlers``."""

from fastapi import APIRouter

from precisionai.agritune.api.config import get_api_root
from precisionai.agritune.api.paths import resolve_under_root
from precisionai.agritune.api.schemas import (
    FeaturesBuildRequest,
    FeaturesBuildResponse,
    FeaturesCleanResponse,
    FeaturesInspectResponse,
    FeaturesStoreRequest,
    FeaturesVerifyResponse,
)
from precisionai.agritune.augmentations.image.pipeline import AugmentationPipelineConfig, ImageAugmentationPipeline
from precisionai.agritune.cli.config import load_augmentation_selection
from precisionai.agritune.features.integrity import verify_store
from precisionai.agritune.features.manifest import FeatureManifest
from precisionai.agritune.features.store import DirectoryFeatureStore
from precisionai.agritune.services.encoder_selection import build_encoder
from precisionai.agritune.services.feature_service import build_features

router = APIRouter(prefix="/features", tags=["features"])


@router.post("/build", response_model=FeaturesBuildResponse)
async def build(request: FeaturesBuildRequest) -> FeaturesBuildResponse:
    """Resumable offline feature precomputation for every sample in a manifest."""
    root = get_api_root()
    manifest_path = resolve_under_root(root, request.manifest_path, field_name="manifest_path")
    store_path = resolve_under_root(root, request.store, field_name="store")
    augmentation_config_path = (
        resolve_under_root(root, request.augmentation_config_path, field_name="augmentation_config_path")
        if request.augmentation_config_path is not None
        else None
    )
    store = DirectoryFeatureStore(store_path)
    encoder, fingerprint = build_encoder(
        base_url=request.encoder.base_url,
        api_key=request.encoder.api_key,
        model=request.encoder.model,
        preprocessing=request.encoder.preprocessing,
    )
    augmentation = load_augmentation_selection(
        str(augmentation_config_path) if augmentation_config_path is not None else None
    )
    pipeline = ImageAugmentationPipeline(
        AugmentationPipelineConfig(geometric=augmentation.geometric, photometric=augmentation.photometric)
    )
    stats = await build_features(
        str(manifest_path),
        store=store,
        encoder=encoder,
        encoder_fingerprint=fingerprint,
        augmentation_mode=augmentation.mode,
        augmentation_pipeline=pipeline,
        global_seed=request.seed,
        augmentation_variant=augmentation.variant,
    )
    return FeaturesBuildResponse(
        total=stats.total,
        computed=stats.computed,
        skipped=stats.skipped,
        failed=stats.failed,
        failed_sample_ids=stats.failed_sample_ids,
    )


@router.post("/verify", response_model=FeaturesVerifyResponse)
def verify(request: FeaturesStoreRequest) -> FeaturesVerifyResponse:
    """Verify every entry in a feature store's checksum against its actual file contents."""
    store_path = resolve_under_root(get_api_root(), request.store, field_name="store")
    report = verify_store(DirectoryFeatureStore(store_path))
    return FeaturesVerifyResponse(is_valid=report.is_valid, total=report.total, corrupted_keys=report.corrupted_keys)


@router.get("/inspect", response_model=FeaturesInspectResponse)
def inspect(store: str) -> FeaturesInspectResponse:
    """Report feature store statistics: entry count, encoder models seen, patch dimensions seen."""
    store_path = resolve_under_root(get_api_root(), store, field_name="store")
    manifest = FeatureManifest.from_store(DirectoryFeatureStore(store_path))
    return FeaturesInspectResponse(
        total_entries=len(manifest),
        encoder_models=dict(manifest.encoder_models()),
        patch_dims=dict(manifest.patch_dims()),
    )


@router.post("/clean", response_model=FeaturesCleanResponse)
def clean(request: FeaturesStoreRequest) -> FeaturesCleanResponse:
    """Remove any tensor/meta file in a feature store whose pair is missing."""
    store_path = resolve_under_root(get_api_root(), request.store, field_name="store")
    removed = DirectoryFeatureStore(store_path).clean()
    return FeaturesCleanResponse(removed=removed)
