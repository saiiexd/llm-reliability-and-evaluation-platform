"""
Reliability diagnostics and forensic claim inspection endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from llm_reliability.api.dependencies import get_diagnostics_service, get_repository
from llm_reliability.api.errors import ResourceNotFoundError
from llm_reliability.api.schemas.common import PaginatedResponse, PaginationMeta
from llm_reliability.api.schemas.diagnostics import ReliabilityDiagnosticResponse
from llm_reliability.api.services import DiagnosticsService, to_diagnostic_response
from llm_reliability.experiments.repository import ExperimentRepository, RunNotFoundError

router = APIRouter(prefix="/diagnostics", tags=["Diagnostics"])


@router.get("", response_model=PaginatedResponse[ReliabilityDiagnosticResponse])
def list_diagnostics(
    run_id: str | None = Query(default=None),
    experiment_id: str | None = Query(default=None),
    test_case_id: str | None = Query(default=None),
    failure_category: str | None = Query(default=None),
    secondary_finding: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    repository: ExperimentRepository = Depends(get_repository),
    diag_service: DiagnosticsService = Depends(get_diagnostics_service),
) -> PaginatedResponse[ReliabilityDiagnosticResponse]:
    """Retrieves reliability diagnostics with category and secondary finding filters."""
    target_run_ids = [run_id] if run_id else repository.list_run_ids(experiment_id=experiment_id)

    matched: list[ReliabilityDiagnosticResponse] = []
    for r_id in target_run_ids:
        try:
            run = repository.load_run(r_id)
        except Exception:
            continue

        diags = diag_service.get_diagnostics(run)
        for tc_id, d in diags.items():
            if test_case_id and tc_id != test_case_id:
                continue
            cat_str = (
                d.failure_category.value
                if (d.failure_category is not None and hasattr(d.failure_category, "value"))
                else str(d.failure_category or "")
            )
            if failure_category and cat_str.lower() != failure_category.lower():
                continue
            if secondary_finding:
                has_finding = any(
                    secondary_finding.lower() in f.lower() for f in d.secondary_findings
                )
                if not has_finding:
                    continue
            matched.append(to_diagnostic_response(d, run_id=r_id))

    total = len(matched)
    items = matched[offset : offset + limit]
    return PaginatedResponse(
        items=items,
        pagination=PaginationMeta(
            total=total,
            limit=limit,
            offset=offset,
            has_next=(offset + limit) < total,
            has_previous=offset > 0,
        ),
    )


@router.get("/{run_id}/{test_case_id}", response_model=ReliabilityDiagnosticResponse)
def get_diagnostic(
    run_id: str,
    test_case_id: str,
    repository: ExperimentRepository = Depends(get_repository),
    diag_service: DiagnosticsService = Depends(get_diagnostics_service),
) -> ReliabilityDiagnosticResponse:
    """Retrieves forensic reliability diagnostic for a specific test case in a run."""
    try:
        run = repository.load_run(run_id)
    except RunNotFoundError as err:
        raise ResourceNotFoundError(resource_type="Run", resource_id=run_id) from err

    diags = diag_service.get_diagnostics(run)
    diag = diags.get(test_case_id)
    if diag is None:
        raise ResourceNotFoundError(resource_type="Diagnostic", resource_id=test_case_id)

    return to_diagnostic_response(diag, run_id=run_id)
