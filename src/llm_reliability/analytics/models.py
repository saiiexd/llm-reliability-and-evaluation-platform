"""
Analytical domain models for the LLM Reliability & Evaluation Platform.

Defines structured records for evaluation aggregation, failure distributions,
retrieval summaries, evaluator agreement, robustness consistency, run comparisons,
trends, and execution performance. All models preserve denominators and provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from llm_reliability.diagnostics.taxonomy import FailureCategory


class ComparisonCompatibility(StrEnum):
    """Categorical compatibility level for comparing multiple experiment runs.

    ``IDENTICAL_CONFIGURATION``: all compared runs share the exact same
    reproducibility configuration fingerprint (identical dataset content,
    model configuration, and evaluator configurations). Represents repeated
    executions to observe stochasticity and consistency.

    ``PARTIALLY_DIFFERENT_CONFIGURATION``: runs share the identical dataset
    version ID, but differ in model parameters, prompt templates, or evaluator
    configurations. Represents controlled experimental interventions.

    ``MATERIALLY_INCOMPATIBLE``: runs operate on different dataset version
    IDs or disjoint evaluation criteria. Direct metric comparison is not
    methodologically valid; differences must be highlighted rather than
    presented as equivalent.
    """

    IDENTICAL_CONFIGURATION = "identical_configuration"
    PARTIALLY_DIFFERENT_CONFIGURATION = "partially_different_configuration"
    MATERIALLY_INCOMPATIBLE = "materially_incompatible"


@dataclass
class EvaluationAggregate:
    """Aggregate statistics for one evaluator and criterion across an execution.

    Preserves exact numerators and denominators for every rate calculation.
    """

    evaluator_name: str
    criterion: str | None
    total_cases: int
    num_evaluated: int
    num_success: int
    num_skipped: int
    num_invalid_config: int
    num_execution_error: int
    num_output_validation_error: int
    num_passed: int
    num_failed: int
    pass_rate: float | None
    pass_rate_numerator: int
    pass_rate_denominator: int
    score_count: int
    score_mean: float | None
    score_min: float | None
    score_max: float | None
    criterion_outcomes: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator_name": self.evaluator_name,
            "criterion": self.criterion,
            "total_cases": self.total_cases,
            "num_evaluated": self.num_evaluated,
            "num_success": self.num_success,
            "num_skipped": self.num_skipped,
            "num_invalid_config": self.num_invalid_config,
            "num_execution_error": self.num_execution_error,
            "num_output_validation_error": self.num_output_validation_error,
            "num_passed": self.num_passed,
            "num_failed": self.num_failed,
            "pass_rate": self.pass_rate,
            "pass_rate_numerator": self.pass_rate_numerator,
            "pass_rate_denominator": self.pass_rate_denominator,
            "score_count": self.score_count,
            "score_mean": self.score_mean,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "criterion_outcomes": self.criterion_outcomes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationAggregate:
        return cls(
            evaluator_name=data["evaluator_name"],
            criterion=data.get("criterion"),
            total_cases=data["total_cases"],
            num_evaluated=data["num_evaluated"],
            num_success=data["num_success"],
            num_skipped=data["num_skipped"],
            num_invalid_config=data["num_invalid_config"],
            num_execution_error=data["num_execution_error"],
            num_output_validation_error=data["num_output_validation_error"],
            num_passed=data["num_passed"],
            num_failed=data["num_failed"],
            pass_rate=data.get("pass_rate"),
            pass_rate_numerator=data.get("pass_rate_numerator", data["num_passed"]),
            pass_rate_denominator=data.get(
                "pass_rate_denominator", data["num_passed"] + data["num_failed"]
            ),
            score_count=data["score_count"],
            score_mean=data.get("score_mean"),
            score_min=data.get("score_min"),
            score_max=data.get("score_max"),
            criterion_outcomes=dict(data.get("criterion_outcomes", {})),
        )


@dataclass
class FailureCategoryCount:
    """Frequency and proportion of a single diagnostic failure category."""

    category: FailureCategory
    count: int
    proportion: float | None
    denominator: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "count": self.count,
            "proportion": self.proportion,
            "denominator": self.denominator,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FailureCategoryCount:
        return cls(
            category=FailureCategory(data["category"]),
            count=data["count"],
            proportion=data.get("proportion"),
            denominator=data["denominator"],
        )


@dataclass
class SecondaryFindingCount:
    """Frequency of a co-occurring secondary diagnostic observation."""

    finding_name: str
    count: int
    frequency: float | None
    denominator: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_name": self.finding_name,
            "count": self.count,
            "frequency": self.frequency,
            "denominator": self.denominator,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SecondaryFindingCount:
        return cls(
            finding_name=data["finding_name"],
            count=data["count"],
            frequency=data.get("frequency"),
            denominator=data["denominator"],
        )


@dataclass
class FailureDistribution:
    """Consolidated distribution of primary failure categories and secondary findings."""

    total_cases: int
    num_determined: int
    num_undetermined: int
    primary_categories: dict[str, FailureCategoryCount]
    secondary_findings: dict[str, SecondaryFindingCount]
    source_run_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "num_determined": self.num_determined,
            "num_undetermined": self.num_undetermined,
            "primary_categories": {k: v.to_dict() for k, v in self.primary_categories.items()},
            "secondary_findings": {k: v.to_dict() for k, v in self.secondary_findings.items()},
            "source_run_ids": self.source_run_ids,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FailureDistribution:
        return cls(
            total_cases=data["total_cases"],
            num_determined=data["num_determined"],
            num_undetermined=data["num_undetermined"],
            primary_categories={
                k: FailureCategoryCount.from_dict(v)
                for k, v in data.get("primary_categories", {}).items()
            },
            secondary_findings={
                k: SecondaryFindingCount.from_dict(v)
                for k, v in data.get("secondary_findings", {}).items()
            },
            source_run_ids=list(data.get("source_run_ids", [])),
        )


@dataclass
class RetrievalSummary:
    """Aggregate retrieval metric preserving K value and ground-truth sample counts."""

    metric_name: str
    k: int | None
    total_cases: int
    evaluated_cases: int
    skipped_cases: int
    mean_score: float | None
    min_score: float | None
    max_score: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "k": self.k,
            "total_cases": self.total_cases,
            "evaluated_cases": self.evaluated_cases,
            "skipped_cases": self.skipped_cases,
            "mean_score": self.mean_score,
            "min_score": self.min_score,
            "max_score": self.max_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrievalSummary:
        return cls(
            metric_name=data["metric_name"],
            k=data.get("k"),
            total_cases=data["total_cases"],
            evaluated_cases=data["evaluated_cases"],
            skipped_cases=data["skipped_cases"],
            mean_score=data.get("mean_score"),
            min_score=data.get("min_score"),
            max_score=data.get("max_score"),
        )


@dataclass
class EvaluatorAgreementSummary:
    """Pairwise agreement between two evaluators on compatible criteria."""

    evaluator_a: str
    evaluator_b: str
    criterion: str
    cases_compared: int
    agreements: int
    disagreements: int
    agreement_rate: float | None
    excluded_cases: int
    disagreement_test_case_ids: list[str] = field(default_factory=list)
    # 2x2 contingency table over the compared cases (a = evaluator_a, b = evaluator_b).
    both_passed: int = 0
    only_a_passed: int = 0
    only_b_passed: int = 0
    both_failed: int = 0
    # Cohen's kappa: agreement corrected for the agreement expected by chance
    # given each evaluator's own pass rate. None when undefined (no cases, or
    # chance agreement is exactly 1, e.g. both evaluators always say "fail").
    cohens_kappa: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "both_passed": self.both_passed,
            "only_a_passed": self.only_a_passed,
            "only_b_passed": self.only_b_passed,
            "both_failed": self.both_failed,
            "cohens_kappa": self.cohens_kappa,
            "evaluator_a": self.evaluator_a,
            "evaluator_b": self.evaluator_b,
            "criterion": self.criterion,
            "cases_compared": self.cases_compared,
            "agreements": self.agreements,
            "disagreements": self.disagreements,
            "agreement_rate": self.agreement_rate,
            "excluded_cases": self.excluded_cases,
            "disagreement_test_case_ids": self.disagreement_test_case_ids,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluatorAgreementSummary:
        return cls(
            evaluator_a=data["evaluator_a"],
            evaluator_b=data["evaluator_b"],
            criterion=data["criterion"],
            cases_compared=data["cases_compared"],
            agreements=data["agreements"],
            disagreements=data["disagreements"],
            agreement_rate=data.get("agreement_rate"),
            excluded_cases=data["excluded_cases"],
            disagreement_test_case_ids=list(data.get("disagreement_test_case_ids", [])),
            both_passed=int(data.get("both_passed", 0)),
            only_a_passed=int(data.get("only_a_passed", 0)),
            only_b_passed=int(data.get("only_b_passed", 0)),
            both_failed=int(data.get("both_failed", 0)),
            cohens_kappa=data.get("cohens_kappa"),
        )


@dataclass
class RobustnessSummary:
    """Consistency analysis across controlled input perturbations."""

    total_pairs: int
    num_unchanged: int
    num_changed: int
    consistency_rate: float | None
    by_perturbation: dict[str, dict[str, Any]] = field(default_factory=dict)
    transition_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_pairs": self.total_pairs,
            "num_unchanged": self.num_unchanged,
            "num_changed": self.num_changed,
            "consistency_rate": self.consistency_rate,
            "by_perturbation": self.by_perturbation,
            "transition_counts": self.transition_counts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RobustnessSummary:
        return cls(
            total_pairs=data["total_pairs"],
            num_unchanged=data["num_unchanged"],
            num_changed=data["num_changed"],
            consistency_rate=data.get("consistency_rate"),
            by_perturbation=dict(data.get("by_perturbation", {})),
            transition_counts=dict(data.get("transition_counts", {})),
        )


@dataclass
class ExecutionPerformanceSummary:
    """Execution timing and latency measurements for one run.

    Explicitly reports missing/unsupported telemetry as unavailable.
    """

    run_id: str
    wall_clock_duration_seconds: float | None
    per_case_duration_seconds_mean: float | None = None
    retrieval_duration_seconds_mean: float | None = None
    generation_duration_seconds_mean: float | None = None
    evaluator_duration_seconds_mean: float | None = None
    tokens_available: bool = False
    cost_available: bool = False
    notes: list[str] = field(default_factory=list)
    prompt_tokens_total: int | None = None
    completion_tokens_total: int | None = None
    total_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "wall_clock_duration_seconds": self.wall_clock_duration_seconds,
            "per_case_duration_seconds_mean": self.per_case_duration_seconds_mean,
            "retrieval_duration_seconds_mean": self.retrieval_duration_seconds_mean,
            "generation_duration_seconds_mean": self.generation_duration_seconds_mean,
            "evaluator_duration_seconds_mean": self.evaluator_duration_seconds_mean,
            "tokens_available": self.tokens_available,
            "cost_available": self.cost_available,
            "notes": self.notes,
            "prompt_tokens_total": self.prompt_tokens_total,
            "completion_tokens_total": self.completion_tokens_total,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionPerformanceSummary:
        return cls(
            run_id=data["run_id"],
            wall_clock_duration_seconds=data.get("wall_clock_duration_seconds"),
            per_case_duration_seconds_mean=data.get("per_case_duration_seconds_mean"),
            retrieval_duration_seconds_mean=data.get("retrieval_duration_seconds_mean"),
            generation_duration_seconds_mean=data.get("generation_duration_seconds_mean"),
            evaluator_duration_seconds_mean=data.get("evaluator_duration_seconds_mean"),
            tokens_available=data.get("tokens_available", False),
            cost_available=data.get("cost_available", False),
            notes=list(data.get("notes", [])),
            prompt_tokens_total=data.get("prompt_tokens_total"),
            completion_tokens_total=data.get("completion_tokens_total"),
            total_tokens=data.get("total_tokens"),
        )


@dataclass
class RunComparison:
    """Side-by-side comparison across two or more runs."""

    source_run_ids: list[str]
    compatibility: ComparisonCompatibility
    compatibility_reasons: list[str]
    configuration_diffs: dict[str, Any]
    evaluation_comparisons: dict[str, dict[str, Any]]
    failure_distribution_comparisons: dict[str, dict[str, int]]
    retrieval_comparisons: dict[str, dict[str, float | None]]
    robustness_comparisons: dict[str, float | None]
    performance_comparisons: dict[str, float | None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_run_ids": self.source_run_ids,
            "compatibility": self.compatibility.value,
            "compatibility_reasons": self.compatibility_reasons,
            "configuration_diffs": self.configuration_diffs,
            "evaluation_comparisons": self.evaluation_comparisons,
            "failure_distribution_comparisons": self.failure_distribution_comparisons,
            "retrieval_comparisons": self.retrieval_comparisons,
            "robustness_comparisons": self.robustness_comparisons,
            "performance_comparisons": self.performance_comparisons,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunComparison:
        return cls(
            source_run_ids=list(data["source_run_ids"]),
            compatibility=ComparisonCompatibility(data["compatibility"]),
            compatibility_reasons=list(data.get("compatibility_reasons", [])),
            configuration_diffs=dict(data.get("configuration_diffs", {})),
            evaluation_comparisons=dict(data.get("evaluation_comparisons", {})),
            failure_distribution_comparisons=dict(data.get("failure_distribution_comparisons", {})),
            retrieval_comparisons=dict(data.get("retrieval_comparisons", {})),
            robustness_comparisons=dict(data.get("robustness_comparisons", {})),
            performance_comparisons=dict(data.get("performance_comparisons", {})),
        )


@dataclass
class TrendAnalysis:
    """Chronological trajectory of metrics across runs."""

    run_ids: list[str]
    experiment_id: str | None
    is_repeated_run_trend: bool
    has_trend: bool
    configuration_transitions: list[int]
    metric_series: dict[str, list[float | None]]
    sample_sizes: list[int]
    timestamps: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_ids": self.run_ids,
            "experiment_id": self.experiment_id,
            "is_repeated_run_trend": self.is_repeated_run_trend,
            "has_trend": self.has_trend,
            "configuration_transitions": self.configuration_transitions,
            "metric_series": self.metric_series,
            "sample_sizes": self.sample_sizes,
            "timestamps": self.timestamps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrendAnalysis:
        return cls(
            run_ids=list(data["run_ids"]),
            experiment_id=data.get("experiment_id"),
            is_repeated_run_trend=data["is_repeated_run_trend"],
            has_trend=data["has_trend"],
            configuration_transitions=list(data.get("configuration_transitions", [])),
            metric_series=dict(data.get("metric_series", {})),
            sample_sizes=list(data.get("sample_sizes", [])),
            timestamps=list(data.get("timestamps", [])),
        )


@dataclass
class AnalyticsOverview:
    """Platform-wide analytical roll-up preserving denominators and data availability."""

    total_experiments: int
    total_runs: int
    total_test_cases: int
    total_evaluations: int
    evaluator_coverage: dict[str, int]
    disagreement_count: int
    primary_failure_distribution: dict[str, int]
    retrieval_runs_count: int
    mean_retrieval_mrr: float | None
    recent_runs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_experiments": self.total_experiments,
            "total_runs": self.total_runs,
            "total_test_cases": self.total_test_cases,
            "total_evaluations": self.total_evaluations,
            "evaluator_coverage": self.evaluator_coverage,
            "disagreement_count": self.disagreement_count,
            "primary_failure_distribution": self.primary_failure_distribution,
            "retrieval_runs_count": self.retrieval_runs_count,
            "mean_retrieval_mrr": self.mean_retrieval_mrr,
            "recent_runs": self.recent_runs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalyticsOverview:
        return cls(
            total_experiments=data["total_experiments"],
            total_runs=data["total_runs"],
            total_test_cases=data["total_test_cases"],
            total_evaluations=data["total_evaluations"],
            evaluator_coverage=dict(data.get("evaluator_coverage", {})),
            disagreement_count=data["disagreement_count"],
            primary_failure_distribution=dict(data.get("primary_failure_distribution", {})),
            retrieval_runs_count=data["retrieval_runs_count"],
            mean_retrieval_mrr=data.get("mean_retrieval_mrr"),
            recent_runs=list(data.get("recent_runs", [])),
        )
