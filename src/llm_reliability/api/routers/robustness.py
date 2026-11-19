"""
Robustness perturbation and consistency evaluation endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from llm_reliability.api.dependencies import get_diagnostics_service, get_repository
from llm_reliability.api.errors import ResourceNotFoundError
from llm_reliability.api.schemas.common import PaginatedResponse, PaginationMeta
from llm_reliability.api.schemas.robustness import (
    RobustnessBundleResponse,
    RobustnessComparisonResponse,
)
from llm_reliability.api.services import (
    DiagnosticsService,
    to_robustness_bundle,
)
from llm_reliability.experiments.repository import ExperimentRepository, RunNotFoundError

router = APIRouter(prefix="/robustness", tags=["Robustness"])


@router.get("", response_model=PaginatedResponse[RobustnessComparisonResponse])
def list_robustness(
    run_id: str | None = Query(default=None),
    perturbation_type: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    repository: ExperimentRepository = Depends(get_repository),
    diag_service: DiagnosticsService = Depends(get_diagnostics_service),
) -> PaginatedResponse[RobustnessComparisonResponse]:
    """Lists perturbation pair comparisons across runs."""
    target_run_ids = [run_id] if run_id else repository.list_run_ids()

    items: list[RobustnessComparisonResponse] = []
    for r_id in target_run_ids:
        try:
            run = repository.load_run(r_id)
        except Exception:
            continue

        bundle = to_robustness_bundle(r_id, diag_service.get_robustness(run))
        for comp in bundle.comparisons:
            if perturbation_type and comp.perturbation_type.lower() != perturbation_type.lower():
                continue
            items.append(comp)

    total = len(items)
    page_items = items[offset : offset + limit]
    return PaginatedResponse(
        items=page_items,
        pagination=PaginationMeta(
            total=total,
            limit=limit,
            offset=offset,
            has_next=(offset + limit) < total,
            has_previous=offset > 0,
        ),
    )


@router.get("/{run_id}", response_model=RobustnessBundleResponse)
def get_run_robustness(
    run_id: str,
    repository: ExperimentRepository = Depends(get_repository),
    diag_service: DiagnosticsService = Depends(get_diagnostics_service),
) -> RobustnessBundleResponse:
    """Retrieves full robustness bundle (comparisons and consistency summary) for a run."""
    try:
        run = repository.load_run(run_id)
    except RunNotFoundError as err:
        raise ResourceNotFoundError(resource_type="Run", resource_id=run_id) from err

    return to_robustness_bundle(run_id, diag_service.get_robustness(run))
