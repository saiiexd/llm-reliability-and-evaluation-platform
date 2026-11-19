"""
Experiment management and controlled benchmark execution endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, status

from llm_reliability.api.dependencies import (
    get_adapter_registry,
    get_evaluator_registry,
    get_experiment_manager,
    get_repository,
)
from llm_reliability.api.errors import (
    ConflictError,
    ExecutionError,
    ResourceNotFoundError,
)
from llm_reliability.api.registry import AdapterRegistry, EvaluatorRegistry
from llm_reliability.api.schemas.common import PaginatedResponse, PaginationMeta
from llm_reliability.api.schemas.experiments import (
    CreateExperimentRequest,
    ExecuteExperimentRequest,
    ExperimentDetailResponse,
    ExperimentSummaryResponse,
    RunExecutionResponse,
)
from llm_reliability.api.schemas.runs import RunSummaryResponse
from llm_reliability.api.services import (
    to_experiment_detail,
    to_experiment_summary,
    to_run_summary,
)
from llm_reliability.evaluation.models import Dataset, ModelConfig, TestCase
from llm_reliability.experiments.manager import ExperimentManager
from llm_reliability.experiments.repository import (
    ExperimentAlreadyExistsError,
    ExperimentNotFoundError,
    ExperimentRepository,
)

router = APIRouter(prefix="/experiments", tags=["Experiments"])


@router.get("", response_model=PaginatedResponse[ExperimentSummaryResponse])
def list_experiments(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    search: str | None = Query(default=None),
    repository: ExperimentRepository = Depends(get_repository),
) -> PaginatedResponse[ExperimentSummaryResponse]:
    """Lists experiment definitions with search and pagination."""
    all_ids = repository.list_experiment_ids()
    experiments: list[ExperimentSummaryResponse] = []

    for exp_id in all_ids:
        try:
            defn = repository.load_experiment(exp_id)
            if (
                search
                and search.lower() not in defn.name.lower()
                and search.lower() not in exp_id.lower()
            ):
                continue
            runs_count = len(repository.list_run_ids(experiment_id=exp_id))
            experiments.append(to_experiment_summary(defn, runs_count=runs_count))
        except Exception:
            continue

    total = len(experiments)
    items = experiments[offset : offset + limit]
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


@router.post("", response_model=ExperimentDetailResponse, status_code=status.HTTP_201_CREATED)
def create_experiment(
    req: CreateExperimentRequest,
    manager: ExperimentManager = Depends(get_experiment_manager),
    eval_registry: EvaluatorRegistry = Depends(get_evaluator_registry),
) -> ExperimentDetailResponse:
    """Safely creates a new experiment definition with allowlisted evaluators."""
    try:
        evaluators = [eval_registry.get(name) for name in req.evaluator_names]
    except ValueError as exc:
        raise ExecutionError(str(exc)) from exc

    dataset = Dataset(
        name=req.dataset_name,
        test_cases=[
            TestCase(
                id=tc.id,
                input=tc.input,
                reference_answer=tc.reference_answer,
                metadata=tc.metadata,
            )
            for tc in req.test_cases
        ],
    )

    model_config = ModelConfig(
        model_id=req.model_config_req.model_id,
        temperature=req.model_config_req.temperature,
        max_tokens=req.model_config_req.max_tokens,
        extra_params=req.model_config_req.extra_params or req.model_config_req.metadata,
    )

    try:
        defn = manager.create_experiment(
            experiment_id=req.experiment_id,
            name=req.name,
            dataset=dataset,
            model_config=model_config,
            evaluators=evaluators,
            description=req.description,
            metadata=req.metadata,
        )
        return to_experiment_detail(defn, runs_count=0)
    except ExperimentAlreadyExistsError as exc:
        raise ConflictError(str(exc)) from exc


@router.get("/{experiment_id}", response_model=ExperimentDetailResponse)
def get_experiment(
    experiment_id: str,
    repository: ExperimentRepository = Depends(get_repository),
) -> ExperimentDetailResponse:
    """Retrieves detailed experiment definition and configuration."""
    try:
        defn = repository.load_experiment(experiment_id)
        runs_count = len(repository.list_run_ids(experiment_id=experiment_id))
        return to_experiment_detail(defn, runs_count=runs_count)
    except ExperimentNotFoundError as err:
        raise ResourceNotFoundError(resource_type="Experiment", resource_id=experiment_id) from err


@router.get("/{experiment_id}/runs", response_model=PaginatedResponse[RunSummaryResponse])
def list_experiment_runs(
    experiment_id: str,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    repository: ExperimentRepository = Depends(get_repository),
) -> PaginatedResponse[RunSummaryResponse]:
    """Lists executed runs for a specific experiment."""
    # Verify experiment exists
    try:
        repository.load_experiment(experiment_id)
    except ExperimentNotFoundError as err:
        raise ResourceNotFoundError(resource_type="Experiment", resource_id=experiment_id) from err

    run_ids = repository.list_run_ids(experiment_id=experiment_id)
    runs: list[RunSummaryResponse] = []
    for r_id in run_ids:
        try:
            run = repository.load_run(r_id)
            runs.append(to_run_summary(run))
        except Exception:
            continue

    total = len(runs)
    items = runs[offset : offset + limit]
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


@router.post(
    "/{experiment_id}/execute",
    response_model=RunExecutionResponse,
    status_code=status.HTTP_201_CREATED,
)
def execute_experiment(
    experiment_id: str,
    req: ExecuteExperimentRequest,
    manager: ExperimentManager = Depends(get_experiment_manager),
    adapter_registry: AdapterRegistry = Depends(get_adapter_registry),
    eval_registry: EvaluatorRegistry = Depends(get_evaluator_registry),
) -> RunExecutionResponse:
    """Executes an experiment using an allowlisted adapter, persisting the result."""
    try:
        defn = manager.load_experiment(experiment_id)
    except ExperimentNotFoundError as err:
        raise ResourceNotFoundError(resource_type="Experiment", resource_id=experiment_id) from err

    if adapter_registry.is_external_provider(req.adapter_id) and not req.allow_external_inference:
        raise ExecutionError(
            f"Adapter '{req.adapter_id}' requires external inference. "
            "Explicit opt-in is required via 'allow_external_inference=True' "
            "to permit outbound network requests."
        )

    try:
        adapter = adapter_registry.get(
            req.adapter_id,
            config={"allow_external_inference": req.allow_external_inference},
        )
    except ValueError as exc:
        raise ExecutionError(str(exc)) from exc

    # Reconstruct the declared evaluators from registry
    try:
        evaluators = [eval_registry.get(ec.evaluator_name) for ec in defn.evaluator_configs]
    except ValueError as exc:
        raise ExecutionError(
            f"Configured evaluator could not be resolved from allowlisted registry: {exc}"
        ) from exc

    try:
        run = manager.execute_experiment(
            definition=defn,
            adapter=adapter,
            evaluators=evaluators,
            run_id=req.run_id,
        )
    except Exception as exc:
        raise ExecutionError(f"Execution failed: {exc}") from exc

    dur = None
    if run.started_at and run.finished_at:
        try:
            from datetime import datetime

            t0 = datetime.fromisoformat(run.started_at)
            t1 = datetime.fromisoformat(run.finished_at)
            dur = max(0.0, (t1 - t0).total_seconds())
        except Exception:
            pass

    return RunExecutionResponse(
        run_id=run.run_id,
        experiment_id=run.experiment_id,
        status=run.status.value,
        started_at=run.started_at,
        finished_at=run.finished_at,
        duration_seconds=dur,
        num_succeeded=run.result.num_succeeded if run.result else 0,
        num_failed=run.result.num_failed if run.result else 0,
        num_passed=run.result.num_passed if run.result else 0,
        pass_rate=run.result.pass_rate if run.result else 0.0,
        error=run.error,
    )
