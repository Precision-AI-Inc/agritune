# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Routes for ``agritune dataset validate``/``inspect`` — see ``precisionai.agritune.cli.handlers``."""

from fastapi import APIRouter

from precisionai.agritune.api.config import get_api_root
from precisionai.agritune.api.paths import resolve_under_root
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
    manifest_path = resolve_under_root(get_api_root(), request.manifest_path, field_name="manifest_path")
    report = validate_dataset(str(manifest_path), num_classes=request.num_classes, ignore_index=request.ignore_index)
    issues = [
        ValidationIssueResponse(sample_id=issue.sample_id, category=issue.category, message=issue.message)
        for issue in report.issues
    ]
    return DatasetValidateResponse(is_valid=report.is_valid, issues=issues)


@router.get("/inspect", response_model=DatasetInspectResponse)
def inspect(manifest_path: str) -> DatasetInspectResponse:
    """Report dataset statistics: sample count, metadata columns, class pixel counts."""
    resolved = resolve_under_root(get_api_root(), manifest_path, field_name="manifest_path")
    return DatasetInspectResponse(**inspect_dataset(str(resolved)))
