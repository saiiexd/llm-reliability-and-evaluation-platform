"""
Pydantic schemas for cross-experiment analytics, agreement, and comparisons.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from llm_reliability.api.schemas.common import BaseApiSchema


class EvaluationAggregateResponse(BaseApiSchema):
    evaluator_name: str
    criterion: str
    total_cases: int
    num_evaluated: int
    num_success: int
    num_skipped: int = 0
    num_invalid_config: int = 0
    num_execution_error: int = 0
    num_output_validation_error: int = 0
    num_passed: int
    num_failed: int
    pass_rate: float | None = None
    pass_rate_numerator: int = 0
    pass_rate_denominator: int = 0
    score_count: int = 0
    score_mean: float | None = None
    score_min: float | None = None
    score_max: float | None = None
    criterion_outcomes: dict[str, int] = Field(default_factory=dict)


class EvaluatorAgreementSummaryResponse(BaseApiSchema):
    evaluator_a: str
    evaluator_b: str
    criterion: str
    cases_compared: int
    agreements: int
    disagreements: int
    agreement_rate: float | None = None
    excluded_cases: int = 0
    disagreement_test_case_ids: list[str] = Field(default_factory=list)
    both_passed: int = 0
    only_a_passed: int = 0
    only_b_passed: int = 0
    both_failed: int = 0
    cohens_kappa: float | None = None


class EvaluatorAnalyticsResponse(BaseApiSchema):
    run_id: str
    aggregates: dict[str, EvaluationAggregateResponse]
    agreements: list[EvaluatorAgreementSummaryResponse] = Field(default_factory=list)


class FailureDistributionResponse(BaseApiSchema):
    run_id: str | None = None
    total_cases: int
    num_determined: int = 0
    num_undetermined: int = 0
    primary_categories: dict[str, Any] = Field(default_factory=dict)
    secondary_findings: dict[str, Any] = Field(default_factory=dict)
    source_run_ids: list[str] = Field(default_factory=list)
    total_test_cases: int = 0
    primary_category_counts: dict[str, int] = Field(default_factory=dict)
    primary_category_proportions: dict[str, float] = Field(default_factory=dict)
    secondary_finding_counts: dict[str, int] = Field(default_factory=dict)
    secondary_finding_proportions: dict[str, float] = Field(default_factory=dict)


class RetrievalSummaryResponse(BaseApiSchema):
    metric_name: str
    k: int | None = None
    total_cases: int = 0
    evaluated_cases: int = 0
    skipped_cases: int = 0
    excluded_cases: int = 0
    mean_score: float | None = None
    score_mean: float | None = None
    min_score: float | None = None
    score_min: float | None = None
    max_score: float | None = None
    score_max: float | None = None


class RobustnessSummaryResponse(BaseApiSchema):
    total_perturbation_pairs: int = 0
    identical_category_count: int = 0
    consistency_rate: float = 0.0
    consistency_by_perturbation: dict[str, float] = Field(default_factory=dict)
    category_transitions: dict[str, int] = Field(default_factory=dict)


class ExecutionPerformanceResponse(BaseApiSchema):
    run_id: str
    wall_clock_duration_seconds: float = 0.0
    wall_clock_seconds: float = 0.0
    per_case_duration_seconds_mean: float | None = None
    mean_test_case_duration_seconds: float | None = None
    retrieval_duration_seconds_mean: float | None = None
    generation_duration_seconds_mean: float | None = None
    evaluator_duration_seconds_mean: float | None = None
    durations_instrumented: bool = False
    tokens_available: bool = False
    cost_available: bool = False
    notes: list[str] = Field(default_factory=list)


class RunComparisonResponse(BaseApiSchema):
    source_run_ids: list[str]
    compatibility: str
    compatibility_reasons: list[str] = Field(default_factory=list)
    configuration_diffs: dict[str, Any] = Field(default_factory=dict)
    evaluation_comparisons: dict[str, Any] = Field(default_factory=dict)
    failure_distribution_comparisons: dict[str, Any] = Field(default_factory=dict)
    retrieval_comparisons: dict[str, Any] = Field(default_factory=dict)
    robustness_comparisons: dict[str, Any] = Field(default_factory=dict)
    performance_comparisons: dict[str, Any] = Field(default_factory=dict)


class TrendAnalysisResponse(BaseApiSchema):
    experiment_id: str | None = None
    run_ids: list[str]
    timestamps: list[str]
    sample_sizes: list[int]
    metric_series: dict[str, list[float | None]]
    configuration_transitions: list[int]
    is_repeated_run_trend: bool
    has_trend: bool


class AnalyticsOverviewResponse(BaseApiSchema):
    total_experiments: int
    total_runs: int
    total_test_cases: int
    total_evaluations: int
    evaluator_coverage: dict[str, int]
    disagreement_count: int
    primary_failure_distribution: dict[str, int]
    retrieval_runs_count: int
    mean_retrieval_mrr: float | None = None
    recent_runs: list[dict[str, Any]] = Field(default_factory=list)
