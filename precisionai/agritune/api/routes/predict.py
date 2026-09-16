# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Route for ``agritune predict`` — see ``precisionai.agritune.cli.handlers``."""

from fastapi import APIRouter

from precisionai.agritune.api.config import get_api_root
from precisionai.agritune.api.paths import resolve_under_root
from precisionai.agritune.api.schemas import PredictRequest, PredictResponse
from precisionai.agritune.features.keys import EncoderFingerprint
from precisionai.agritune.features.store import build_feature_store
from precisionai.agritune.services.prediction_service import PredictionRunConfig, run_prediction

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Run inference and write one prediction PNG (and optional overlay) per sample."""
    root = get_api_root()
    manifest_path = resolve_under_root(root, request.manifest_path, field_name="manifest_path")
    store_path = resolve_under_root(root, request.store, field_name="store")
    checkpoint_path = resolve_under_root(root, request.checkpoint_path, field_name="checkpoint_path")
    output_dir = resolve_under_root(root, request.output_dir, field_name="output_dir")
    store = build_feature_store(request.store_type, store_path, entries_per_shard=request.entries_per_shard)
    config = PredictionRunConfig(
        manifest_path=str(manifest_path),
        checkpoint_path=str(checkpoint_path),
        output_dir=str(output_dir),
        num_classes=request.num_classes,
        encoder_fingerprint=EncoderFingerprint(
            model=request.encoder_fingerprint.model,
            revision=request.encoder_fingerprint.revision,
            preprocessing=request.encoder_fingerprint.preprocessing,
        ),
        decoder_name=request.decoder,
        batch_size=request.batch_size,
        sample_ids=request.sample_ids,
        write_overlays=request.overlays,
        overlay_alpha=request.overlay_alpha,
    )
    return PredictResponse(written=run_prediction(config, store=store))
