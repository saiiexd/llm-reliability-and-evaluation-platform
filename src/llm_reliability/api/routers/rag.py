"""
RAG retrieval trace and evaluation endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from llm_reliability.api.dependencies import get_diagnostics_service, get_repository
from llm_reliability.api.errors import ResourceNotFoundError
from llm_reliability.api.schemas.common import PaginatedResponse, PaginationMeta
from llm_reliability.api.schemas.rag import RagTestCaseResultResponse
from llm_reliability.api.services import DiagnosticsService, to_rag_response
from llm_reliability.experiments.repository import ExperimentRepository, RunNotFoundError

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.get("/retrieval-results", response_model=PaginatedResponse[RagTestCaseResultResponse])
def list_retrieval_results(
    run_id: str | None = Query(default=None),
    test_case_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    repository: ExperimentRepository = Depends(get_repository),
    diag_service: DiagnosticsService = Depends(get_diagnostics_service),
) -> PaginatedResponse[RagTestCaseResultResponse]:
    """Retrieves paginated RAG retrieval traces and retrieval evaluations."""
    target_run_ids = [run_id] if run_id else repository.list_run_ids()

    items: list[RagTestCaseResultResponse] = []
    for r_id in target_run_ids:
        try:
            run = repository.load_run(r_id)
        except Exception:
            continue

        if not run.result:
            continue

        ret_results = diag_service.get_retrieval_results(run)
        if not ret_results:
            continue

        for tc in run.result.test_case_results:
            if test_case_id and tc.test_case_id != test_case_id:
                continue
            case_evals = ret_results.get(tc.test_case_id, [])
            items.append(to_rag_response(tc.test_case_id, r_id, tc, case_evals))

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


@router.get(
    "/retrieval-results/{run_id}/{test_case_id}",
    response_model=RagTestCaseResultResponse,
)
def get_retrieval_result(
    run_id: str,
    test_case_id: str,
    repository: ExperimentRepository = Depends(get_repository),
    diag_service: DiagnosticsService = Depends(get_diagnostics_service),
) -> RagTestCaseResultResponse:
    """Retrieves RAG trace and retrieval metrics for a specific test case."""
    try:
        run = repository.load_run(run_id)
    except RunNotFoundError as err:
        raise ResourceNotFoundError(resource_type="Run", resource_id=run_id) from err

    if not run.result:
        raise ResourceNotFoundError(resource_type="TestCase", resource_id=test_case_id)

    tc_res = next(
        (tc for tc in run.result.test_case_results if tc.test_case_id == test_case_id),
        None,
    )
    if tc_res is None:
        raise ResourceNotFoundError(resource_type="TestCase", resource_id=test_case_id)

    ret_results = diag_service.get_retrieval_results(run)
    case_evals = ret_results.get(test_case_id, [])
    return to_rag_response(test_case_id, run_id, tc_res, case_evals)
