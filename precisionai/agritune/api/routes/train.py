# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Route for ``agritune train`` — see ``precisionai.agritune.cli.handlers``."""

from fastapi import APIRouter

from precisionai.agritune.api.schemas import TrainRequest, TrainResponse
from precisionai.agritune.cli.config import load_training_run_config
from precisionai.agritune.features.store import DirectoryFeatureStore
from precisionai.agritune.services.training_service import run_training

router = APIRouter(tags=["train"])


@router.post("/train", response_model=TrainResponse)
def train(request: TrainRequest) -> TrainResponse:
    """Run one full offline training job from a YAML config, with optional dotlist overrides."""
    config = load_training_run_config(request.config_path, request.overrides)
    store = DirectoryFeatureStore(config.feature_store_dir)
    result = run_training(config, store=store)
    return TrainResponse(
        run_directory=str(result.run_directory.path),
        final_epoch=result.final_train_state["epoch"],
        global_optimizer_step=result.final_train_state["global_optimizer_step"],
        best_metric=result.final_train_state["best_metric"],
        train_metrics=result.train_metrics,
        val_metrics=result.val_metrics,
    )
