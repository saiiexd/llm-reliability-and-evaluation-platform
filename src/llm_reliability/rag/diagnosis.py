"""
RAG failure diagnosis: an evidence-based classification, not a guess.

This module answers a narrower question than "why is this answer wrong":
given only the evaluation evidence already computed (a retrieval hit/miss
signal from ground-truth-based retrieval evaluation, and a correctness
signal from answer evaluation), which of a small set of categories is
consistent with that evidence? When the evidence does not clearly support
one category, the result is explicitly ``UNDETERMINED`` -- this is a
correct and expected outcome, not a failure of the diagnosis, and callers
must not treat it as a coin-flip default.

Decision table (evidence -> category), the entire logic of this module::

    retrieval_hit | answer_correct | category
    --------------|-----------------|----------------------------------------------
    None          | any             | RETRIEVAL_EVIDENCE_UNAVAILABLE
    True          | True            | SUCCESSFUL_GROUNDED_EXECUTION
    True          | False           | GENERATION_FAILURE_DESPITE_RELEVANT_CONTEXT
    True          | None            | UNDETERMINED
    False         | False           | RETRIEVAL_FAILURE
    False         | True            | UNDETERMINED

``retrieval_hit`` is ``None`` whenever no ground-truth-based retrieval
evaluator produced a ``SUCCESS`` result (no ground truth was available, or
the response carried no retrieval evidence at all): the diagnosis cannot
attribute a failure to retrieval or clear retrieval of any wrongdoing
without knowing whether it actually succeeded, so it reports
``RETRIEVAL_EVIDENCE_UNAVAILABLE`` regardless of the answer's correctness.

``retrieval_hit is False`` but the answer is nonetheless correct
(``answer_correct is True``) is deliberately ``UNDETERMINED``, not treated
as a success: the required evidence was not surfaced by retrieval (a real
retrieval shortcoming), yet the generation step produced a correct answer
anyway (for example, from the model's own parametric knowledge). This
platform does not have a category for that combination and will not
force it into either "success" or "failure" -- it is genuinely
ambiguous given only these two signals.

An incorrect answer is never, by itself, classified as retrieval failure:
that requires the additional evidence that a known-relevant chunk was
*not* retrieved. Likewise, a correct retrieval hit is never enough by
itself to certify "successful grounded execution" without also knowing
the answer was actually correct.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from llm_reliability.evaluation.criteria import CORRECTNESS
from llm_reliability.evaluation.models import EvaluationResult, EvaluationStatus
from llm_reliability.rag.evaluators import RetrievalEvaluationResult, RetrievalEvaluationStatus


class RagDiagnosisCategory(StrEnum):
    RETRIEVAL_FAILURE = "retrieval_failure"
    RETRIEVAL_EVIDENCE_UNAVAILABLE = "retrieval_evidence_unavailable"
    GENERATION_FAILURE_DESPITE_RELEVANT_CONTEXT = "generation_failure_despite_relevant_context"
    SUCCESSFUL_GROUNDED_EXECUTION = "successful_grounded_execution"
    UNDETERMINED = "undetermined"


@dataclass
class RagDiagnosis:
    """One test case's evidence-based RAG failure classification.

    ``evidence`` preserves exactly the signals the classification was
    based on, so a human can verify or challenge the category rather than
    trusting it blindly.
    """

    test_case_id: str
    category: RagDiagnosisCategory
    explanation: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_case_id": self.test_case_id,
            "category": self.category.value,
            "explanation": self.explanation,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RagDiagnosis:
        return cls(
            test_case_id=data["test_case_id"],
            category=RagDiagnosisCategory(data["category"]),
            explanation=data["explanation"],
            evidence=dict(data.get("evidence", {})),
        )


def _aggregate_bool_evidence(values: list[bool]) -> bool | None:
    """Combine boolean signals: True if all True, False if all False, else None (disagreement)."""
    if not values:
        return None
    if all(values):
        return True
    if not any(values):
        return False
    return None


def _retrieval_hit_evidence(retrieval_results: list[RetrievalEvaluationResult]) -> bool | None:
    hit_scores = [
        r.score > 0.0
        for r in retrieval_results
        if r.status == RetrievalEvaluationStatus.SUCCESS
        and r.metric_name in ("hit_at_k", "mrr")
        and r.score is not None
    ]
    return _aggregate_bool_evidence(hit_scores)


def _answer_correct_evidence(answer_results: list[EvaluationResult]) -> bool | None:
    correctness_judgments = [
        bool(r.passed)
        for r in answer_results
        if r.status == EvaluationStatus.SUCCESS
        and r.criterion == CORRECTNESS
        and r.passed is not None
    ]
    return _aggregate_bool_evidence(correctness_judgments)


def diagnose_rag_execution(
    test_case_id: str,
    retrieval_results: list[RetrievalEvaluationResult],
    answer_results: list[EvaluationResult],
) -> RagDiagnosis:
    """Classify one test case's RAG execution using only the evidence given.

    ``retrieval_results`` should be the retrieval evaluator results for
    this test case (see ``llm_reliability.rag.evaluators.evaluate_retrieval``);
    ``answer_results`` should be its ``TestCaseResult.evaluation_results``
    (answer-quality results, from ``llm_reliability.evaluation``). Neither
    list is re-executed or mutated.
    """
    retrieval_hit = _retrieval_hit_evidence(retrieval_results)
    answer_correct = _answer_correct_evidence(answer_results)
    evidence = {"retrieval_hit": retrieval_hit, "answer_correct": answer_correct}

    if retrieval_hit is None:
        return RagDiagnosis(
            test_case_id=test_case_id,
            category=RagDiagnosisCategory.RETRIEVAL_EVIDENCE_UNAVAILABLE,
            explanation=(
                "No ground-truth-based retrieval evaluation evidence is available for this "
                "test case, so retrieval cannot be ruled in or out as a cause."
            ),
            evidence=evidence,
        )

    if retrieval_hit is True and answer_correct is True:
        return RagDiagnosis(
            test_case_id=test_case_id,
            category=RagDiagnosisCategory.SUCCESSFUL_GROUNDED_EXECUTION,
            explanation="A relevant chunk was retrieved and the answer was judged correct.",
            evidence=evidence,
        )

    if retrieval_hit is True and answer_correct is False:
        return RagDiagnosis(
            test_case_id=test_case_id,
            category=RagDiagnosisCategory.GENERATION_FAILURE_DESPITE_RELEVANT_CONTEXT,
            explanation=(
                "A relevant chunk was retrieved, but the answer was judged incorrect: the "
                "evidence is consistent with a generation/context-use failure, not a "
                "retrieval failure."
            ),
            evidence=evidence,
        )

    if retrieval_hit is False and answer_correct is False:
        return RagDiagnosis(
            test_case_id=test_case_id,
            category=RagDiagnosisCategory.RETRIEVAL_FAILURE,
            explanation=(
                "No relevant chunk was retrieved, and the answer was judged incorrect: the "
                "evidence is consistent with a retrieval failure."
            ),
            evidence=evidence,
        )

    return RagDiagnosis(
        test_case_id=test_case_id,
        category=RagDiagnosisCategory.UNDETERMINED,
        explanation=(
            "The available evidence does not clearly support any single category "
            f"(retrieval_hit={retrieval_hit!r}, answer_correct={answer_correct!r}); reporting "
            "undetermined rather than guessing."
        ),
        evidence=evidence,
    )
