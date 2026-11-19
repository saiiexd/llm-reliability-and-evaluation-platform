"""
Analytics and observability package for LLM Reliability & Evaluation.

Provides structured aggregation, cross-run comparison, evaluator agreement,
failure taxonomy analysis, retrieval metrics, and longitudinal trends.
"""

from __future__ import annotations

from llm_reliability.analytics.aggregations import (
    aggregate_evaluation,
    aggregate_failures,
    aggregate_retrieval,
    aggregate_robustness,
    measure_performance,
)
from llm_reliability.analytics.agreement import (
    compute_all_pairwise_agreements,
    compute_evaluator_agreement,
)
from llm_reliability.analytics.comparison import assess_compatibility, compare_runs
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
from llm_reliability.analytics.service import AnalyticsService
from llm_reliability.analytics.trends import compute_trend_analysis

__all__ = [
    "AnalyticsOverview",
    "AnalyticsService",
    "ComparisonCompatibility",
    "EvaluationAggregate",
    "EvaluatorAgreementSummary",
    "ExecutionPerformanceSummary",
    "FailureCategoryCount",
    "FailureDistribution",
    "RetrievalSummary",
    "RobustnessSummary",
    "RunComparison",
    "SecondaryFindingCount",
    "TrendAnalysis",
    "aggregate_evaluation",
    "aggregate_failures",
    "aggregate_retrieval",
    "aggregate_robustness",
    "assess_compatibility",
    "compare_runs",
    "compute_all_pairwise_agreements",
    "compute_evaluator_agreement",
    "compute_trend_analysis",
    "measure_performance",
]
