"""
Reliability findings: per-test-case flags derived from existing evaluation evidence.

This module classifies evaluation *evidence*, not model behavior. It never
determines whether a generated answer is factually true, and it never
labels an answer a hallucination. "Hallucination" is a specific, stronger
claim -- an assertion unsupported by, or contradicting, available context
-- that requires context-grounded evaluation; that is explicitly out of
scope here and deferred to a later milestone (see
``research/notes/evaluation_and_reliability_layer.md``).

What this module can honestly say is narrower, and fully grounded in the
evaluation results already computed: whether evaluators agreed with each
other, whether any evaluation evidence was even available, whether an
evaluator failed to run, and whether a specific reference-based evaluator
reported a mismatch (a fact about that evaluator's methodology, not a
determination that the answer is fabricated).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from llm_reliability.evaluation.models import EvaluationStatus, RunResult, TestCaseResult

_ERROR_STATUSES = (
    EvaluationStatus.EXECUTION_ERROR,
    EvaluationStatus.OUTPUT_VALIDATION_ERROR,
    EvaluationStatus.INVALID_CONFIGURATION,
)


class ReliabilityFlagType(StrEnum):
    """Categories of reliability findings this layer can currently identify.

    None of these categories asserts that an answer is factually wrong or
    fabricated ("hallucinated"); they describe properties of the
    evaluation evidence itself, not of ground truth.
    """

    EVALUATOR_DISAGREEMENT = "evaluator_disagreement"
    MISSING_EVALUATION_EVIDENCE = "missing_evaluation_evidence"
    EVALUATOR_FAILURE = "evaluator_failure"
    INCORRECT_PER_EVALUATOR = "incorrect_per_evaluator"


@dataclass
class ReliabilityFinding:
    """One flagged observation about the evaluation evidence for a single test case."""

    test_case_id: str
    flag_type: ReliabilityFlagType
    detail: str
    evaluator_names: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_case_id": self.test_case_id,
            "flag_type": self.flag_type.value,
            "detail": self.detail,
            "evaluator_names": list(self.evaluator_names),
        }


def analyze_reliability(run_result: RunResult) -> list[ReliabilityFinding]:
    """Classify the evaluation evidence already present in ``run_result``.

    Read-only: never re-runs evaluators or the model, and never mutates
    ``run_result``. Test cases whose generation itself failed are skipped
    here, since that is already fully captured by
    ``TestCaseResult.execution_error`` and is not an evaluator reliability
    question.
    """
    findings: list[ReliabilityFinding] = []
    for tc_result in run_result.test_case_results:
        if not tc_result.succeeded:
            continue
        findings.extend(_findings_for_test_case(tc_result))
    return findings


def _findings_for_test_case(tc_result: TestCaseResult) -> list[ReliabilityFinding]:
    results = tc_result.evaluation_results
    findings: list[ReliabilityFinding] = []

    failed_evaluators = tuple(r.evaluator_name for r in results if r.status in _ERROR_STATUSES)
    if failed_evaluators:
        findings.append(
            ReliabilityFinding(
                test_case_id=tc_result.test_case_id,
                flag_type=ReliabilityFlagType.EVALUATOR_FAILURE,
                detail=f"{len(failed_evaluators)} evaluator(s) failed to produce a result.",
                evaluator_names=failed_evaluators,
            )
        )

    if results and all(r.status == EvaluationStatus.SKIPPED for r in results):
        findings.append(
            ReliabilityFinding(
                test_case_id=tc_result.test_case_id,
                flag_type=ReliabilityFlagType.MISSING_EVALUATION_EVIDENCE,
                detail="No evaluator was able to produce a judgment for this test case.",
                evaluator_names=tuple(r.evaluator_name for r in results),
            )
        )

    successful = [
        r for r in results if r.status == EvaluationStatus.SUCCESS and r.passed is not None
    ]
    passed_values = {r.passed for r in successful}
    if len(passed_values) > 1:
        findings.append(
            ReliabilityFinding(
                test_case_id=tc_result.test_case_id,
                flag_type=ReliabilityFlagType.EVALUATOR_DISAGREEMENT,
                detail="Evaluators disagreed on whether the answer passed for this test case.",
                evaluator_names=tuple(r.evaluator_name for r in successful),
            )
        )

    incorrect_evaluators = tuple(r.evaluator_name for r in successful if r.passed is False)
    if incorrect_evaluators:
        findings.append(
            ReliabilityFinding(
                test_case_id=tc_result.test_case_id,
                flag_type=ReliabilityFlagType.INCORRECT_PER_EVALUATOR,
                detail=(
                    "Reported as incorrect by the listed evaluator(s), relative to their own "
                    "methodology. This is not a hallucination determination."
                ),
                evaluator_names=incorrect_evaluators,
            )
        )

    return findings
