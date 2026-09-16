"""
Evaluation summary: aggregate statistics over a RunResult's evaluation results.

Aggregation never replaces or discards the individual per-test-case,
per-evaluator ``EvaluationResult`` objects; ``RunResult.test_case_results``
remains the source of truth and is untouched by this module. Every
aggregate number here is scoped explicitly to one evaluator (and the
criterion it assesses), never collapsed into a single, ungrounded overall
score for the experiment: an experiment run can use several evaluators
that disagree, and this summary is designed to make that visible rather
than average it away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_reliability.evaluation.models import EvaluationStatus, RunResult, TestCaseResult

_ERROR_STATUSES = (
    EvaluationStatus.EXECUTION_ERROR,
    EvaluationStatus.OUTPUT_VALIDATION_ERROR,
    EvaluationStatus.INVALID_CONFIGURATION,
)


@dataclass
class EvaluatorSummary:
    """Aggregate statistics for one evaluator across a run.

    ``num_evaluated`` counts every test case this evaluator was applied to
    (i.e., every test case whose generation succeeded); the remaining
    ``num_*`` counts partition that total by ``EvaluationStatus``.
    ``score_mean``/``score_min``/``score_max`` are computed only over
    results that actually reported a numeric score (``score is not
    None``); an evaluator that reports purely categorical outcomes, such
    as ``LLMJudgeEvaluator``, will have ``score_count == 0`` and all three
    left ``None`` rather than a misleading zero.
    """

    evaluator_name: str
    criterion: str | None
    num_evaluated: int
    num_success: int
    num_skipped: int
    num_error: int
    num_passed: int
    num_failed: int
    score_count: int
    score_mean: float | None
    score_min: float | None
    score_max: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator_name": self.evaluator_name,
            "criterion": self.criterion,
            "num_evaluated": self.num_evaluated,
            "num_success": self.num_success,
            "num_skipped": self.num_skipped,
            "num_error": self.num_error,
            "num_passed": self.num_passed,
            "num_failed": self.num_failed,
            "score_count": self.score_count,
            "score_mean": self.score_mean,
            "score_min": self.score_min,
            "score_max": self.score_max,
        }


@dataclass
class EvaluationSummary:
    """Aggregate statistics for a full run, broken out per evaluator.

    Deliberately has no single "overall score" field: ``per_evaluator``
    keeps every evaluator's statistics separate and explicitly labeled, so
    reading this summary never requires (or permits) collapsing several
    evaluation methods into one number.
    """

    dataset_name: str
    model_id: str
    num_test_cases: int
    num_execution_succeeded: int
    num_execution_failed: int
    per_evaluator: dict[str, EvaluatorSummary]
    disagreement_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "model_id": self.model_id,
            "num_test_cases": self.num_test_cases,
            "num_execution_succeeded": self.num_execution_succeeded,
            "num_execution_failed": self.num_execution_failed,
            "per_evaluator": {name: s.to_dict() for name, s in self.per_evaluator.items()},
            "disagreement_count": self.disagreement_count,
        }


def summarize_run(run_result: RunResult) -> EvaluationSummary:
    """Compute an ``EvaluationSummary`` from an existing ``RunResult``.

    Read-only: never re-runs evaluators or the model, and never mutates
    ``run_result``.
    """
    per_evaluator_results: dict[str, list] = {}
    for tc_result in run_result.test_case_results:
        for ev_result in tc_result.evaluation_results:
            per_evaluator_results.setdefault(ev_result.evaluator_name, []).append(ev_result)

    per_evaluator: dict[str, EvaluatorSummary] = {}
    for evaluator_name, results in per_evaluator_results.items():
        criterion = next((r.criterion for r in results if r.criterion is not None), None)
        num_success = sum(1 for r in results if r.status == EvaluationStatus.SUCCESS)
        num_skipped = sum(1 for r in results if r.status == EvaluationStatus.SKIPPED)
        num_error = sum(1 for r in results if r.status in _ERROR_STATUSES)
        num_passed = sum(
            1 for r in results if r.status == EvaluationStatus.SUCCESS and r.passed is True
        )
        num_failed = sum(
            1 for r in results if r.status == EvaluationStatus.SUCCESS and r.passed is False
        )
        scores = [r.score for r in results if r.score is not None]
        per_evaluator[evaluator_name] = EvaluatorSummary(
            evaluator_name=evaluator_name,
            criterion=criterion,
            num_evaluated=len(results),
            num_success=num_success,
            num_skipped=num_skipped,
            num_error=num_error,
            num_passed=num_passed,
            num_failed=num_failed,
            score_count=len(scores),
            score_mean=(sum(scores) / len(scores)) if scores else None,
            score_min=min(scores) if scores else None,
            score_max=max(scores) if scores else None,
        )

    disagreement_count = sum(
        1 for tc_result in run_result.test_case_results if _has_disagreement(tc_result)
    )

    return EvaluationSummary(
        dataset_name=run_result.dataset_name,
        model_id=run_result.model_id,
        num_test_cases=run_result.num_test_cases,
        num_execution_succeeded=run_result.num_succeeded,
        num_execution_failed=run_result.num_failed,
        per_evaluator=per_evaluator,
        disagreement_count=disagreement_count,
    )


def _has_disagreement(tc_result: TestCaseResult) -> bool:
    passed_values = {
        r.passed
        for r in tc_result.evaluation_results
        if r.status == EvaluationStatus.SUCCESS and r.passed is not None
    }
    return len(passed_values) > 1
