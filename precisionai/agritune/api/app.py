# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""``create_app()`` — the FastAPI application factory.

Registers every route module from :mod:`precisionai.agritune.api.routes` and a handful of
exception handlers translating the domain exceptions services already raise into HTTP status
codes: ``ValueError`` (bad input, e.g. an empty sample set) to 400, a missing manifest/checkpoint
file to 404, an uncached feature lookup to 404, and a checkpoint fingerprint mismatch to 409. No
other translation happens — an unexpected exception still surfaces as a 500, same as it would
reach an uncaught traceback from the CLI.
"""

import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from precisionai.agritune.api.routes import (
    dataset_router,
    encoder_router,
    evaluate_router,
    features_router,
    predict_router,
    train_router,
)
from precisionai.agritune.features.errors import FeatureNotCachedError
from precisionai.agritune.logging import configure_logging
from precisionai.agritune.training.checkpointing import CheckpointMismatchError


def create_app() -> FastAPI:
    """Construct the ``agritune`` FastAPI application.

    Every route delegates to :mod:`precisionai.agritune.services` — the exact same functions the
    CLI calls — so no business logic is duplicated between entry points.

    Returns
    -------
    fastapi.FastAPI
    """
    configure_logging(os.environ.get("AGRITUNE_LOG_LEVEL", "INFO"))
    app = FastAPI(
        title="AgriTune API",
        description="Train and evaluate agricultural segmentation decoders on frozen features "
        "from a remote ViT encoder.",
    )

    app.include_router(dataset_router)
    app.include_router(features_router)
    app.include_router(train_router)
    app.include_router(evaluate_router)
    app.include_router(predict_router)
    app.include_router(encoder_router)

    @app.exception_handler(ValueError)
    async def handle_value_error(_request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(FileNotFoundError)
    async def handle_file_not_found(_request: Request, exc: FileNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(FeatureNotCachedError)
    async def handle_feature_not_cached(_request: Request, exc: FeatureNotCachedError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(CheckpointMismatchError)
    async def handle_checkpoint_mismatch(_request: Request, exc: CheckpointMismatchError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    return app
