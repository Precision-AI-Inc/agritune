# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Route for ``agritune evaluate`` — see ``precisionai.agritune.cli.handlers``."""

from fastapi import APIRouter

from precisionai.agritune.api.config import get_api_root
from precisionai.agritune.api.paths import resolve_under_root
from precisionai.agritune.api.schemas import EvaluateRequest, EvaluateResponse
from precisionai.agritune.features.keys import EncoderFingerprint
from precisionai.agritune.features.store import DirectoryFeatureStore
from precisionai.agritune.services.evaluation_service import EvaluationRunConfig, run_evaluation
from precisionai.agritune.tasks.segmentation.losses import SegmentationLossConfig

router = APIRouter(tags=["evaluate"])


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(request: EvaluateRequest) -> EvaluateResponse:
    """Load a checkpointed decoder and score it against a dataset (or a restricted sample set)."""
    root = get_api_root()
    manifest_path = resolve_under_root(root, request.manifest_path, field_name="manifest_path")
    store_path = resolve_under_root(root, request.store, field_name="store")
    checkpoint_path = resolve_under_root(root, request.checkpoint_path, field_name="checkpoint_path")
    store = DirectoryFeatureStore(store_path)
    config = EvaluationRunConfig(
        manifest_path=str(manifest_path),
        checkpoint_path=str(checkpoint_path),
        num_classes=request.num_classes,
        encoder_fingerprint=EncoderFingerprint(
            model=request.encoder_fingerprint.model,
            revision=request.encoder_fingerprint.revision,
            preprocessing=request.encoder_fingerprint.preprocessing,
        ),
        decoder_name=request.decoder,
        batch_size=request.batch_size,
        sample_ids=request.sample_ids,
        loss=SegmentationLossConfig(
            name=request.loss.name,
            ignore_index=request.loss.ignore_index,
            ce_weight=request.loss.ce_weight,
            dice_weight=request.loss.dice_weight,
        ),
        resize=request.resize,
    )
    return EvaluateResponse(metrics=run_evaluation(config, store=store))
