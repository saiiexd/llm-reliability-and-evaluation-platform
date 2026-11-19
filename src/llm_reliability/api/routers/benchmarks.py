"""
Controlled multi-model benchmarking endpoints with repetition semantics.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, status

from llm_reliability.api.dependencies import (
    get_adapter_registry,
    get_evaluator_registry,
    get_experiment_manager,
    get_repository,
)
from llm_reliability.api.errors import ExecutionError, ResourceNotFoundError
from llm_reliability.api.registry import AdapterRegistry, EvaluatorRegistry
from llm_reliability.api.schemas.benchmarks import (
    BenchmarkExecutionResponse,
    ExecuteBenchmarkRequest,
)
from llm_reliability.api.schemas.experiments import RunExecutionResponse
from llm_reliability.api.schemas.runs import RunSummaryResponse
from llm_reliability.api.services import to_run_summary
from llm_reliability.experiments.manager import ExperimentManager
from llm_reliability.experiments.repository import (
    ExperimentNotFoundError,
    ExperimentRepository,
)

router = APIRouter(prefix="/benchmarks", tags=["Benchmarks"])


@router.post(
    "/execute",
    response_model=BenchmarkExecutionResponse,
    status_code=status.HTTP_201_CREATED,
)
def execute_benchmark(
    req: ExecuteBenchmarkRequest,
    manager: ExperimentManager = Depends(get_experiment_manager),
    adapter_registry: AdapterRegistry = Depends(get_adapter_registry),
    eval_registry: EvaluatorRegistry = Depends(get_evaluator_registry),
) -> BenchmarkExecutionResponse:
    """Execute a controlled benchmark across selected adapters with repetitions."""
    try:
        defn = manager.load_experiment(req.experiment_id)
    except ExperimentNotFoundError as err:
        raise ResourceNotFoundError(
            resource_type="Experiment", resource_id=req.experiment_id
        ) from err

    # Validate all adapter IDs exist
    for adapter_id in req.adapter_ids:
        if adapter_id not in adapter_registry.list_available():
            available = ", ".join(adapter_registry.list_available())
            raise ExecutionError(
                f"Unknown adapter '{adapter_id}'. Allowlisted adapters: {available}"
            )
        # Check safety opt-in for external providers
        if adapter_registry.is_external_provider(adapter_id) and not req.allow_external_inference:
            raise ExecutionError(
                f"Adapter '{adapter_id}' requires external inference. "
                "Explicit opt-in is required via 'allow_external_inference=True' "
                "to permit outbound network requests."
            )

    # Reconstruct evaluators
    try:
        evaluators = [eval_registry.get(ec.evaluator_name) for ec in defn.evaluator_configs]
    except ValueError as exc:
        raise ExecutionError(
            f"Configured evaluator could not be resolved from allowlisted registry: {exc}"
        ) from exc

    # Validate every adapter against the definition before executing any, so a
    # mismatch (e.g. a top-1 pipeline on a top-2 experiment) is a clean 400
    # instead of a benchmark whose first adapters' runs were already saved.
    for adapter_id in req.adapter_ids:
        try:
            manager.validate_execution(
                defn,
                adapter_registry.get(
                    adapter_id,
                    config={"allow_external_inference": req.allow_external_inference},
                ),
                evaluators,
            )
        except ValueError as exc:
            raise ExecutionError(
                f"Adapter '{adapter_id}' cannot run this experiment: {exc}"
            ) from exc

    benchmark_id = f"bench-{uuid.uuid4().hex[:8]}"
    run_responses: list[RunExecutionResponse] = []

    for adapter_id in req.adapter_ids:
        for rep in range(1, req.repetitions + 1):
            run_id = f"{benchmark_id}-{adapter_id}-rep{rep}-{uuid.uuid4().hex[:4]}"
            try:
                adapter = adapter_registry.get(
                    adapter_id,
                    config={"allow_external_inference": req.allow_external_inference},
                )
                run = manager.execute_experiment(
                    definition=defn,
                    adapter=adapter,
                    evaluators=evaluators,
                    run_id=run_id,
                    benchmark_id=benchmark_id,
                    repetition_index=rep,
                    adapter_id=adapter_id,
                )
            except Exception as exc:
                raise ExecutionError(
                    f"Execution failed for adapter '{adapter_id}' (repetition {rep}): {exc}"
                ) from exc

            dur: float | None = None
            if run.started_at and run.finished_at:
                try:
                    t0 = datetime.fromisoformat(run.started_at)
                    t1 = datetime.fromisoformat(run.finished_at)
                    dur = max(0.0, (t1 - t0).total_seconds())
                except Exception:
                    pass

            run_responses.append(
                RunExecutionResponse(
                    run_id=run.run_id,
                    experiment_id=run.experiment_id,
                    status=run.status.value,
                    started_at=run.started_at,
                    finished_at=run.finished_at,
                    duration_seconds=dur,
                    num_succeeded=run.result.num_succeeded if run.result else 0,
                    num_failed=run.result.num_failed if run.result else 0,
                    error=run.error,
                )
            )

    return BenchmarkExecutionResponse(
        benchmark_id=benchmark_id,
        experiment_id=req.experiment_id,
        total_runs=len(run_responses),
        runs=run_responses,
    )


@router.get("/{benchmark_id}/runs", response_model=list[RunSummaryResponse])
def get_benchmark_runs(
    benchmark_id: str,
    repository: ExperimentRepository = Depends(get_repository),
) -> list[RunSummaryResponse]:
    """Retrieve all executed runs associated with a specific benchmark ID."""
    all_runs = repository.list_run_ids()
    matched_runs: list[RunSummaryResponse] = []

    for r_id in all_runs:
        try:
            run = repository.load_run(r_id)
            if run.benchmark_id == benchmark_id:
                matched_runs.append(to_run_summary(run))
        except Exception:
            continue

    return matched_runs
