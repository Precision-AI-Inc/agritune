# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Routes for ``agritune dataset validate``/``inspect`` — see ``precisionai.agritune.cli.handlers``."""

from fastapi import APIRouter

from precisionai.agritune.api.schemas import (
    DatasetInspectResponse,
    DatasetValidateRequest,
    DatasetValidateResponse,
    ValidationIssueResponse,
)
from precisionai.agritune.services.dataset_service import inspect_dataset, validate_dataset

router = APIRouter(prefix="/dataset", tags=["dataset"])


@router.post("/validate", response_model=DatasetValidateResponse)
def validate(request: DatasetValidateRequest) -> DatasetValidateResponse:
    """Validate a dataset manifest — missing images/masks, invalid labels, duplicate IDs, ..."""
    report = validate_dataset(request.manifest_path, num_classes=request.num_classes, ignore_index=request.ignore_index)
    issues = [
        ValidationIssueResponse(sample_id=issue.sample_id, category=issue.category, message=issue.message)
        for issue in report.issues
    ]
    return DatasetValidateResponse(is_valid=report.is_valid, issues=issues)


@router.get("/inspect", response_model=DatasetInspectResponse)
def inspect(manifest_path: str) -> DatasetInspectResponse:
    """Report dataset statistics: sample count, metadata columns, class pixel counts."""
    return DatasetInspectResponse(**inspect_dataset(manifest_path))
