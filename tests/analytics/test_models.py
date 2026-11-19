"""
Tests for analytics domain models serialization and deserialization.
"""

from __future__ import annotations

from llm_reliability.analytics.models import (
    AnalyticsOverview,
    ComparisonCompatibility,
    EvaluationAggregate,
    EvaluatorAgreementSummary,
    ExecutionPerformanceSummary,
    FailureCategoryCount,
    FailureDistribution,
    RetrievalSummary,
    RobustnessSummary,
    RunComparison,
    SecondaryFindingCount,
    TrendAnalysis,
)
from llm_reliability.diagnostics.taxonomy import FailureCategory


def test_evaluation_aggregate_serialization() -> None:
    agg = EvaluationAggregate(
        evaluator_name="exact_match",
        criterion="correctness",
        total_cases=10,
        num_evaluated=10,
        num_success=9,
        num_skipped=1,
        num_invalid_config=0,
        num_execution_error=0,
        num_output_validation_error=0,
        num_passed=7,
        num_failed=2,
        pass_rate=7 / 9,
        pass_rate_numerator=7,
        pass_rate_denominator=9,
        score_count=9,
        score_mean=0.777,
        score_min=0.0,
        score_max=1.0,
        criterion_outcomes={"exact_match": 7},
    )
    d = agg.to_dict()
    assert d["evaluator_name"] == "exact_match"
    assert d["pass_rate_numerator"] == 7
    assert d["pass_rate_denominator"] == 9
    restored = EvaluationAggregate.from_dict(d)
    assert restored == agg


def test_failure_distribution_serialization() -> None:
    f_dist = FailureDistribution(
        total_cases=5,
        num_determined=5,
        num_undetermined=0,
        primary_categories={
            "correct": FailureCategoryCount(
                category=FailureCategory.CORRECT, count=3, proportion=0.6, denominator=5
            ),
            "unsupported": FailureCategoryCount(
                category=FailureCategory.UNSUPPORTED, count=2, proportion=0.4, denominator=5
            ),
        },
        secondary_findings={
            "insufficient_evidence": SecondaryFindingCount(
                finding_name="insufficient_evidence", count=1, frequency=0.2, denominator=5
            )
        },
        source_run_ids=["run-1"],
    )
    d = f_dist.to_dict()
    assert d["source_run_ids"] == ["run-1"]
    assert len(d["primary_categories"]) == 2
    assert (
        d["secondary_findings"]["insufficient_evidence"]["finding_name"] == "insufficient_evidence"
    )
    restored = FailureDistribution.from_dict(d)
    assert restored.source_run_ids == f_dist.source_run_ids
    assert len(restored.primary_categories) == 2
    assert restored.primary_categories["correct"].category == FailureCategory.CORRECT


def test_retrieval_summary_serialization() -> None:
    rs = RetrievalSummary(
        metric_name="hit_at_k",
        k=3,
        total_cases=10,
        evaluated_cases=8,
        skipped_cases=2,
        mean_score=0.75,
        min_score=0.0,
        max_score=1.0,
    )
    d = rs.to_dict()
    assert d["metric_name"] == "hit_at_k"
    assert d["k"] == 3
    restored = RetrievalSummary.from_dict(d)
    assert restored == rs


def test_evaluator_agreement_serialization() -> None:
    eas = EvaluatorAgreementSummary(
        evaluator_a="exact_match",
        evaluator_b="llm_judge",
        criterion="correctness",
        cases_compared=10,
        agreements=8,
        disagreements=2,
        agreement_rate=0.8,
        excluded_cases=1,
        disagreement_test_case_ids=["tc-3", "tc-7"],
    )
    d = eas.to_dict()
    assert d["cases_compared"] == 10
    restored = EvaluatorAgreementSummary.from_dict(d)
    assert restored == eas


def test_robustness_summary_serialization() -> None:
    rs = RobustnessSummary(
        total_pairs=5,
        num_unchanged=4,
        num_changed=1,
        consistency_rate=0.8,
        by_perturbation={"typo": {"total": 5, "unchanged": 4, "changed": 1, "rate": 0.8}},
        transition_counts={"CORRECT->INCORRECT": 1},
    )
    d = rs.to_dict()
    restored = RobustnessSummary.from_dict(d)
    assert restored == rs


def test_performance_summary_serialization() -> None:
    perf = ExecutionPerformanceSummary(
        run_id="run-1",
        wall_clock_duration_seconds=1.23,
        per_case_duration_seconds_mean=0.12,
        tokens_available=False,
        cost_available=False,
        notes=["Cost calculation unavailable."],
    )
    d = perf.to_dict()
    restored = ExecutionPerformanceSummary.from_dict(d)
    assert restored == perf


def test_run_comparison_serialization() -> None:
    comp = RunComparison(
        source_run_ids=["run-1", "run-2"],
        compatibility=ComparisonCompatibility.IDENTICAL_CONFIGURATION,
        compatibility_reasons=["Identical config fingerprint."],
        configuration_diffs={"temperature": {"run-1": 0.0, "run-2": 0.0}},
        evaluation_comparisons={"exact_match": {"pass_rates": {"run-1": 1.0, "run-2": 0.9}}},
        failure_distribution_comparisons={"CORRECT": {"run-1": 9, "run-2": 8}},
        retrieval_comparisons={"mrr": {"run-1": 1.0, "run-2": 1.0}},
        robustness_comparisons={"run-1": 0.8, "run-2": 0.85},
        performance_comparisons={"wall_clock_seconds": {"run-1": 2.1, "run-2": 2.0}},
    )
    d = comp.to_dict()
    restored = RunComparison.from_dict(d)
    assert restored.source_run_ids == comp.source_run_ids
    assert restored.compatibility == ComparisonCompatibility.IDENTICAL_CONFIGURATION


def test_trend_analysis_serialization() -> None:
    trend = TrendAnalysis(
        run_ids=["run-1", "run-2"],
        experiment_id="exp-1",
        is_repeated_run_trend=True,
        has_trend=True,
        configuration_transitions=[],
        metric_series={"exact_match_pass_rate": [0.8, 0.9]},
        sample_sizes=[10, 10],
        timestamps=["2026-11-01T10:00:00Z", "2026-11-01T11:00:00Z"],
    )
    d = trend.to_dict()
    restored = TrendAnalysis.from_dict(d)
    assert restored == trend


def test_analytics_overview_serialization() -> None:
    overview = AnalyticsOverview(
        total_experiments=2,
        total_runs=4,
        total_test_cases=40,
        total_evaluations=80,
        evaluator_coverage={"exact_match": 40, "llm_judge": 40},
        disagreement_count=3,
        primary_failure_distribution={"CORRECT": 32, "INCORRECT": 8},
        retrieval_runs_count=2,
        mean_retrieval_mrr=0.88,
        recent_runs=[{"run_id": "run-4", "status": "succeeded"}],
    )
    d = overview.to_dict()
    restored = AnalyticsOverview.from_dict(d)
    assert restored == overview
