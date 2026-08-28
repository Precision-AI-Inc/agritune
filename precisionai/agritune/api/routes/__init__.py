# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0
"""FastAPI route handlers — delegate to ``precisionai.agritune.services``."""

from precisionai.agritune.api.routes.dataset import router as dataset_router
from precisionai.agritune.api.routes.encoder import router as encoder_router
from precisionai.agritune.api.routes.evaluate import router as evaluate_router
from precisionai.agritune.api.routes.features import router as features_router
from precisionai.agritune.api.routes.predict import router as predict_router
from precisionai.agritune.api.routes.train import router as train_router

__all__ = [
    "dataset_router",
    "encoder_router",
    "evaluate_router",
    "features_router",
    "predict_router",
    "train_router",
]
