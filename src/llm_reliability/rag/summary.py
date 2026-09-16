"""
Retrieval evaluation summary: aggregate statistics across test cases.

Mirrors ``llm_reliability.reliability.summary``'s approach for answer
evaluation: read-only aggregation, scoped explicitly per retrieval
evaluator, never collapsed into one number. This is also where "mean" in
Mean Reciprocal Rank is actually computed -- ``MeanReciprocalRankEvaluator``
reports one reciprocal rank per test case, and this module averages it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_reliability.rag.evaluators import RetrievalEvaluationResult, RetrievalEvaluationStatus


@dataclass
class RetrievalEvaluatorSummary:
    """Aggregate statistics for one retrieval evaluator across a set of test cases."""

    evaluator_name: str
    metric_name: str
    num_evaluated: int
    num_success: int
    num_skipped: int
    num_error: int
    score_mean: float | None
    score_min: float | None
    score_max: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator_name": self.evaluator_name,
            "metric_name": self.metric_name,
            "num_evaluated": self.num_evaluated,
            "num_success": self.num_success,
            "num_skipped": self.num_skipped,
            "num_error": self.num_error,
            "score_mean": self.score_mean,
            "score_min": self.score_min,
            "score_max": self.score_max,
        }


def summarize_retrieval(
    results_by_test_case: dict[str, list[RetrievalEvaluationResult]],
) -> dict[str, RetrievalEvaluatorSummary]:
    """Aggregate retrieval evaluation results, keyed by evaluator name.

    Read-only: never mutates its input. Score statistics are computed only
    over ``SUCCESS`` results, since ``SKIPPED``/``EXECUTION_ERROR`` results
    carry no score.
    """
    by_evaluator: dict[str, list[RetrievalEvaluationResult]] = {}
    for test_case_results in results_by_test_case.values():
        for result in test_case_results:
            by_evaluator.setdefault(result.evaluator_name, []).append(result)

    summaries: dict[str, RetrievalEvaluatorSummary] = {}
    for evaluator_name, results in by_evaluator.items():
        metric_name = results[0].metric_name
        num_success = sum(1 for r in results if r.status == RetrievalEvaluationStatus.SUCCESS)
        num_skipped = sum(1 for r in results if r.status == RetrievalEvaluationStatus.SKIPPED)
        num_error = sum(1 for r in results if r.status == RetrievalEvaluationStatus.EXECUTION_ERROR)
        scores = [r.score for r in results if r.score is not None]
        summaries[evaluator_name] = RetrievalEvaluatorSummary(
            evaluator_name=evaluator_name,
            metric_name=metric_name,
            num_evaluated=len(results),
            num_success=num_success,
            num_skipped=num_skipped,
            num_error=num_error,
            score_mean=(sum(scores) / len(scores)) if scores else None,
            score_min=min(scores) if scores else None,
            score_max=max(scores) if scores else None,
        )
    return summaries
