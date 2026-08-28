# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Route for ``agritune predict`` — see ``precisionai.agritune.cli.handlers``."""

from fastapi import APIRouter

from precisionai.agritune.api.schemas import PredictRequest, PredictResponse
from precisionai.agritune.features.keys import EncoderFingerprint
from precisionai.agritune.features.store import DirectoryFeatureStore
from precisionai.agritune.services.prediction_service import PredictionRunConfig, run_prediction

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Run inference and write one prediction PNG (and optional overlay) per sample."""
    store = DirectoryFeatureStore(request.store)
    config = PredictionRunConfig(
        manifest_path=request.manifest_path,
        checkpoint_path=request.checkpoint_path,
        output_dir=request.output_dir,
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
