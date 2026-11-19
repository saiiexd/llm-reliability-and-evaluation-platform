"""
Experiment run inspection and test case detail endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from llm_reliability.api.dependencies import get_diagnostics_service, get_repository
from llm_reliability.api.errors import ResourceNotFoundError
from llm_reliability.api.schemas.common import PaginatedResponse, PaginationMeta
from llm_reliability.api.schemas.runs import (
    RunDetailResponse,
    RunSummariesResponse,
    RunSummaryResponse,
    TestCaseResultResponse,
)
from llm_reliability.api.services import (
    DiagnosticsService,
    to_run_detail,
    to_run_summary,
)
from llm_reliability.experiments.repository import ExperimentRepository, RunNotFoundError

router = APIRouter(prefix="/runs", tags=["Runs"])


@router.get("", response_model=PaginatedResponse[RunSummaryResponse])
def list_runs(
    experiment_id: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    repository: ExperimentRepository = Depends(get_repository),
) -> PaginatedResponse[RunSummaryResponse]:
    """Lists runs across all experiments with filtering and pagination."""
    run_ids = repository.list_run_ids(experiment_id=experiment_id)
    summaries: list[RunSummaryResponse] = []

    for r_id in run_ids:
        try:
            run = repository.load_run(r_id)
            if status and run.status.value != status:
                continue
            summaries.append(to_run_summary(run))
        except Exception:
            continue

    total = len(summaries)
    items = summaries[offset : offset + limit]
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


@router.get("/{run_id}", response_model=RunDetailResponse)
def get_run(
    run_id: str,
    include_test_cases: bool = Query(default=True),
    repository: ExperimentRepository = Depends(get_repository),
) -> RunDetailResponse:
    """Retrieves full execution run details, optionally omitting raw test cases."""
    try:
        run = repository.load_run(run_id)
        return to_run_detail(run, include_test_cases=include_test_cases)
    except RunNotFoundError as err:
        raise ResourceNotFoundError(resource_type="Run", resource_id=run_id) from err


@router.get("/{run_id}/test-cases", response_model=PaginatedResponse[TestCaseResultResponse])
def list_test_cases(
    run_id: str,
    passed: bool | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    repository: ExperimentRepository = Depends(get_repository),
) -> PaginatedResponse[TestCaseResultResponse]:
    """Lists paginated test case evaluations for a run."""
    try:
        run = repository.load_run(run_id)
    except RunNotFoundError as err:
        raise ResourceNotFoundError(resource_type="Run", resource_id=run_id) from err

    full_detail = to_run_detail(run, include_test_cases=True)
    all_cases = full_detail.test_case_results
    if passed is not None:
        all_cases = [c for c in all_cases if c.passed == passed]

    total = len(all_cases)
    items = all_cases[offset : offset + limit]
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


@router.get("/{run_id}/test-cases/{test_case_id}", response_model=TestCaseResultResponse)
def get_test_case(
    run_id: str,
    test_case_id: str,
    repository: ExperimentRepository = Depends(get_repository),
) -> TestCaseResultResponse:
    """Retrieves an individual test case evaluation record."""
    try:
        run = repository.load_run(run_id)
    except RunNotFoundError as err:
        raise ResourceNotFoundError(resource_type="Run", resource_id=run_id) from err

    full_detail = to_run_detail(run, include_test_cases=True)
    for c in full_detail.test_case_results:
        if c.test_case_id == test_case_id:
            return c

    raise ResourceNotFoundError(resource_type="TestCase", resource_id=test_case_id)


@router.get("/{run_id}/summaries", response_model=RunSummariesResponse)
def get_run_summaries(
    run_id: str,
    repository: ExperimentRepository = Depends(get_repository),
    diag_service: DiagnosticsService = Depends(get_diagnostics_service),
) -> RunSummariesResponse:
    """Retrieves evaluation summary, reliability findings, and retrieval summary."""
    try:
        run = repository.load_run(run_id)
        return diag_service.get_run_summaries(run)
    except RunNotFoundError as err:
        raise ResourceNotFoundError(resource_type="Run", resource_id=run_id) from err
