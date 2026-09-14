# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Route for ``agritune evaluate`` — see ``precisionai.agritune.cli.handlers``."""

from fastapi import APIRouter

from precisionai.agritune.api.schemas import EvaluateRequest, EvaluateResponse
from precisionai.agritune.features.keys import EncoderFingerprint
from precisionai.agritune.features.store import DirectoryFeatureStore
from precisionai.agritune.services.evaluation_service import EvaluationRunConfig, run_evaluation
from precisionai.agritune.tasks.segmentation.losses import SegmentationLossConfig

router = APIRouter(tags=["evaluate"])


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(request: EvaluateRequest) -> EvaluateResponse:
    """Load a checkpointed decoder and score it against a dataset (or a restricted sample set)."""
    store = DirectoryFeatureStore(request.store)
    config = EvaluationRunConfig(
        manifest_path=request.manifest_path,
        checkpoint_path=request.checkpoint_path,
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
