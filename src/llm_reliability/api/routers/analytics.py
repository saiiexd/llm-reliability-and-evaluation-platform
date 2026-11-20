"""
Cross-experiment analytical aggregation and comparison endpoints.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from llm_reliability.analytics.service import AnalyticsService
from llm_reliability.api.dependencies import get_analytics_service, get_repository
from llm_reliability.api.errors import ExecutionError, ResourceNotFoundError
from llm_reliability.api.schemas.analytics import (
    AnalyticsOverviewResponse,
    EvaluationAggregateResponse,
    EvaluatorAgreementSummaryResponse,
    EvaluatorAnalyticsResponse,
    ExecutionPerformanceResponse,
    FailureDistributionResponse,
    RepeatConsistencyResponse,
    RepeatGroupConsistencyResponse,
    RetrievalSummaryResponse,
    RobustnessSummaryResponse,
    RunComparisonResponse,
    TrendAnalysisResponse,
)
from llm_reliability.experiments.repository import ExperimentRepository, RunNotFoundError

router = APIRouter(prefix="/analytics", tags=["Analytics"])


def _map_failure_dist(dist: Any, run_id: str | None = None) -> FailureDistributionResponse:
    d = dist.to_dict()
    prim_counts = {k: v["count"] for k, v in d.get("primary_categories", {}).items()}
    prim_props = {k: v["proportion"] for k, v in d.get("primary_categories", {}).items()}
    sec_counts = {k: v["count"] for k, v in d.get("secondary_findings", {}).items()}
    sec_props = {k: v["frequency"] for k, v in d.get("secondary_findings", {}).items()}
    return FailureDistributionResponse(
        run_id=run_id or (d.get("source_run_ids", [None])[0] if d.get("source_run_ids") else None),
        total_cases=d.get("total_cases", 0),
        total_test_cases=d.get("total_cases", 0),
        num_determined=d.get("num_determined", 0),
        num_undetermined=d.get("num_undetermined", 0),
        primary_categories=d.get("primary_categories", {}),
        secondary_findings=d.get("secondary_findings", {}),
        source_run_ids=d.get("source_run_ids", []),
        primary_category_counts=prim_counts,
        primary_category_proportions=prim_props,
        secondary_finding_counts=sec_counts,
        secondary_finding_proportions=sec_props,
    )


def _map_retrieval(s: Any) -> RetrievalSummaryResponse:
    d = s.to_dict()
    return RetrievalSummaryResponse(
        metric_name=d.get("metric_name", ""),
        k=d.get("k"),
        total_cases=d.get("total_cases", 0),
        evaluated_cases=d.get("evaluated_cases", 0),
        skipped_cases=d.get("skipped_cases", 0),
        excluded_cases=d.get("skipped_cases", 0),
        mean_score=d.get("mean_score"),
        score_mean=d.get("mean_score"),
        min_score=d.get("min_score"),
        score_min=d.get("min_score"),
        max_score=d.get("max_score"),
        score_max=d.get("max_score"),
    )


def _map_robustness(s: Any) -> RobustnessSummaryResponse | None:
    if s is None:
        return None
    d = s.to_dict()
    return RobustnessSummaryResponse(
        total_perturbation_pairs=d.get("total_perturbation_pairs", 0),
        identical_category_count=d.get("identical_category_count", 0),
        consistency_rate=d.get("consistency_rate", 0.0),
        consistency_by_perturbation=d.get("consistency_by_perturbation", {}),
        category_transitions=d.get("category_transitions", {}),
    )


def _map_perf(perf: Any) -> ExecutionPerformanceResponse:
    d = perf.to_dict()
    return ExecutionPerformanceResponse(
        run_id=d.get("run_id", ""),
        wall_clock_duration_seconds=d.get("wall_clock_duration_seconds", 0.0),
        wall_clock_seconds=d.get("wall_clock_duration_seconds", 0.0),
        per_case_duration_seconds_mean=d.get("per_case_duration_seconds_mean"),
        mean_test_case_duration_seconds=d.get("per_case_duration_seconds_mean"),
        retrieval_duration_seconds_mean=d.get("retrieval_duration_seconds_mean"),
        generation_duration_seconds_mean=d.get("generation_duration_seconds_mean"),
        evaluator_duration_seconds_mean=d.get("evaluator_duration_seconds_mean"),
        durations_instrumented=d.get("per_case_duration_seconds_mean") is not None,
        tokens_available=d.get("tokens_available", False),
        cost_available=d.get("cost_available", False),
        notes=d.get("notes", []),
    )


@router.get("/overview", response_model=AnalyticsOverviewResponse)
def get_overview(
    analytics: AnalyticsService = Depends(get_analytics_service),
) -> AnalyticsOverviewResponse:
    """Retrieves platform-wide overview statistics across all experiments and runs."""
    overview = analytics.get_overview()
    return AnalyticsOverviewResponse(
        total_experiments=overview.total_experiments,
        total_runs=overview.total_runs,
        total_test_cases=overview.total_test_cases,
        total_evaluations=overview.total_evaluations,
        evaluator_coverage=overview.evaluator_coverage,
        disagreement_count=overview.disagreement_count,
        primary_failure_distribution=overview.primary_failure_distribution,
        retrieval_runs_count=overview.retrieval_runs_count,
        mean_retrieval_mrr=overview.mean_retrieval_mrr,
        recent_runs=overview.recent_runs,
    )


@router.get("/evaluators", response_model=EvaluatorAnalyticsResponse)
def get_evaluator_analytics(
    run_id: str = Query(...),
    analytics: AnalyticsService = Depends(get_analytics_service),
) -> EvaluatorAnalyticsResponse:
    """Retrieves evaluator aggregates and pairwise agreement summaries for a run."""
    try:
        aggs = analytics.get_run_evaluation_aggregates(run_id)
        agreements = analytics.get_run_agreements(run_id)
    except RunNotFoundError as err:
        raise ResourceNotFoundError(resource_type="Run", resource_id=run_id) from err

    agg_responses = {name: EvaluationAggregateResponse(**ea.to_dict()) for name, ea in aggs.items()}
    agreement_responses = [
        EvaluatorAgreementSummaryResponse(
            evaluator_a=a.evaluator_a,
            evaluator_b=a.evaluator_b,
            criterion=a.criterion,
            cases_compared=a.cases_compared,
            agreements=a.agreements,
            disagreements=a.disagreements,
            agreement_rate=a.agreement_rate,
            excluded_cases=a.excluded_cases,
            disagreement_test_case_ids=list(a.disagreement_test_case_ids),
            both_passed=a.both_passed,
            only_a_passed=a.only_a_passed,
            only_b_passed=a.only_b_passed,
            both_failed=a.both_failed,
            cohens_kappa=a.cohens_kappa,
        )
        for a in agreements
    ]
    return EvaluatorAnalyticsResponse(
        run_id=run_id,
        aggregates=agg_responses,
        agreements=agreement_responses,
    )


@router.get("/failures", response_model=FailureDistributionResponse)
def get_failure_analytics(
    run_id: str = Query(...),
    analytics: AnalyticsService = Depends(get_analytics_service),
) -> FailureDistributionResponse:
    """Retrieves primary failure distributions and co-occurring secondary findings."""
    try:
        dist = analytics.get_run_failures(run_id)
    except RunNotFoundError as err:
        raise ResourceNotFoundError(resource_type="Run", resource_id=run_id) from err

    return _map_failure_dist(dist, run_id=run_id)


@router.get("/retrieval", response_model=list[RetrievalSummaryResponse])
def get_retrieval_analytics(
    run_id: str = Query(...),
    analytics: AnalyticsService = Depends(get_analytics_service),
) -> list[RetrievalSummaryResponse]:
    """Retrieves Hit@K, Recall@K, and MRR metrics with preserved K values."""
    try:
        summaries = analytics.get_run_retrieval(run_id)
    except RunNotFoundError as err:
        raise ResourceNotFoundError(resource_type="Run", resource_id=run_id) from err

    return [_map_retrieval(s) for s in summaries]


@router.get("/robustness", response_model=RobustnessSummaryResponse | None)
def get_robustness_analytics(
    run_id: str = Query(...),
    analytics: AnalyticsService = Depends(get_analytics_service),
) -> RobustnessSummaryResponse | None:
    """Retrieves perturbation consistency rate and category transitions."""
    try:
        summary = analytics.get_run_robustness(run_id)
    except RunNotFoundError as err:
        raise ResourceNotFoundError(resource_type="Run", resource_id=run_id) from err

    return _map_robustness(summary)


@router.get("/performance", response_model=ExecutionPerformanceResponse)
def get_performance_analytics(
    run_id: str = Query(...),
    analytics: AnalyticsService = Depends(get_analytics_service),
) -> ExecutionPerformanceResponse:
    """Retrieves duration and latency telemetry, reporting missing tokens/costs explicitly."""
    try:
        perf = analytics.get_run_performance(run_id)
    except RunNotFoundError as err:
        raise ResourceNotFoundError(resource_type="Run", resource_id=run_id) from err

    return _map_perf(perf)


@router.get("/comparisons", response_model=RunComparisonResponse)
def compare_runs_endpoint(
    run_ids: list[str] = Query(..., min_length=2),
    analytics: AnalyticsService = Depends(get_analytics_service),
) -> RunComparisonResponse:
    """Compares two or more runs with strict compatibility categorization."""
    try:
        comparison = analytics.compare(run_ids)
        return RunComparisonResponse(**comparison.to_dict())
    except Exception as exc:
        raise ExecutionError(f"Run comparison failed: {exc}") from exc


@router.get("/trends/{experiment_id}", response_model=TrendAnalysisResponse)
def get_experiment_trends(
    experiment_id: str,
    repository: ExperimentRepository = Depends(get_repository),
    analytics: AnalyticsService = Depends(get_analytics_service),
) -> TrendAnalysisResponse:
    """Retrieves longitudinal trajectory and configuration transition markers for an experiment."""
    try:
        repository.load_experiment(experiment_id)
    except Exception as err:
        raise ResourceNotFoundError(resource_type="Experiment", resource_id=experiment_id) from err

    trend = analytics.get_trend(experiment_id)
    return TrendAnalysisResponse(
        experiment_id=trend.experiment_id,
        run_ids=trend.run_ids,
        timestamps=trend.timestamps,
        sample_sizes=trend.sample_sizes,
        metric_series=trend.metric_series,
        configuration_transitions=trend.configuration_transitions,
        is_repeated_run_trend=trend.is_repeated_run_trend,
        has_trend=trend.has_trend,
    )


@router.get("/consistency", response_model=RepeatConsistencyResponse)
def get_repeat_consistency(
    experiment_id: str | None = Query(None),
    repository: ExperimentRepository = Depends(get_repository),
    analytics: AnalyticsService = Depends(get_analytics_service),
) -> RepeatConsistencyResponse:
    """Repeated-run consistency (RQ4): how much identical repeat runs vary.

    Groups runs sharing experiment, configuration fingerprint and adapter;
    only groups with two or more runs with results are returned.
    """
    if experiment_id is not None:
        try:
            repository.load_experiment(experiment_id)
        except Exception as err:
            raise ResourceNotFoundError(
                resource_type="Experiment", resource_id=experiment_id
            ) from err

    groups = analytics.get_repeat_consistency(experiment_id)
    return RepeatConsistencyResponse(
        experiment_id=experiment_id,
        groups=[RepeatGroupConsistencyResponse(**g.to_dict()) for g in groups],
    )
