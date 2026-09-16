"""
Consolidated reliability diagnosis for one test case.

Where ``llm_reliability.rag.diagnosis.diagnose_rag_execution`` combines
exactly two signals (a retrieval hit and an answer-correctness verdict)
into a RAG-specific category, ``build_reliability_diagnostic`` here
combines every relevant signal available for a test case -- correctness,
faithfulness, and retrieval evidence together -- into one
``FailureCategory`` from the fuller taxonomy in
``llm_reliability.diagnostics.taxonomy``. It does not replace the
per-evaluator results already visible on
``TestCaseResult.evaluation_results``, or the RAG-specific diagnosis; it
synthesizes them into one record while preserving, in
``contributing_evaluators``, exactly which evaluators the synthesis is
based on, so the underlying evidence is never hidden behind the category
label alone.

Precedence used when evidence could support more than one category (most
specific evidence wins):

1. Generation itself failed (``TestCaseResult.execution_error``) -> ``ERROR``.
2. No relevant evaluation evidence at all -> ``INSUFFICIENT_EVIDENCE``.
3. Correctness evaluators disagree with each other -> ``EVALUATOR_DISAGREEMENT``.
4. A faithfulness evaluator reports contradiction -> ``CONTRADICTED``.
5. A faithfulness evaluator reports the answer unsupported -> ``UNSUPPORTED``.
6. Ground-truth retrieval evidence shows a miss and the answer is
   incorrect -> ``RETRIEVAL_FAILURE``.
7. Ground-truth retrieval evidence shows a hit and the answer is
   incorrect -> ``GENERATION_FAILURE``.
8. The answer is judged correct -> ``CORRECT``.
9. The answer is judged incorrect (no retrieval evidence to attribute it further) -> ``INCORRECT``.
10. None of the above resolves cleanly -> ``UNDETERMINED``.

Faithfulness evidence (steps 4-5) is checked before the coarser
retrieval-hit-based categories (steps 6-7) because, when a faithfulness
evaluator has actually examined the answer against the context, its
verdict is more specific and more actionable than "generation failed
somehow despite relevant context": knowing an answer is unsupported or
contradicted says more than knowing it merely failed exact match.

Every step above is evidence a caller can inspect independently; nothing
here invents evidence that was not already produced by an evaluator or a
retrieval evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_reliability.diagnostics.evidence import ClaimAssessment
from llm_reliability.diagnostics.taxonomy import DiagnosticStatus, FailureCategory
from llm_reliability.evaluation.criteria import CORRECTNESS, FAITHFULNESS
from llm_reliability.evaluation.models import EvaluationStatus, TestCase, TestCaseResult
from llm_reliability.rag.evaluators import RetrievalEvaluationResult, RetrievalEvaluationStatus
from llm_reliability.rag.pipeline import extract_retrieved_chunks


@dataclass
class ReliabilityDiagnostic:
    """A consolidated, evidence-grounded reliability diagnosis for one test case."""

    test_case_id: str
    question: str
    generated_answer: str | None
    reference_answer: str | None
    retrieved_context: list[str] | None
    status: DiagnosticStatus
    failure_category: FailureCategory | None
    contributing_evaluators: list[str] = field(default_factory=list)
    supporting_evidence: list[str] = field(default_factory=list)
    confidence: float | None = None
    explanation: str = ""
    claim_assessments: list[ClaimAssessment] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_case_id": self.test_case_id,
            "question": self.question,
            "generated_answer": self.generated_answer,
            "reference_answer": self.reference_answer,
            "retrieved_context": self.retrieved_context,
            "status": self.status.value,
            "failure_category": self.failure_category.value if self.failure_category else None,
            "contributing_evaluators": self.contributing_evaluators,
            "supporting_evidence": self.supporting_evidence,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "claim_assessments": [c.to_dict() for c in self.claim_assessments],
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReliabilityDiagnostic:
        failure_category = data.get("failure_category")
        return cls(
            test_case_id=data["test_case_id"],
            question=data["question"],
            generated_answer=data.get("generated_answer"),
            reference_answer=data.get("reference_answer"),
            retrieved_context=data.get("retrieved_context"),
            status=DiagnosticStatus(data["status"]),
            failure_category=FailureCategory(failure_category) if failure_category else None,
            contributing_evaluators=list(data.get("contributing_evaluators", [])),
            supporting_evidence=list(data.get("supporting_evidence", [])),
            confidence=data.get("confidence"),
            explanation=data.get("explanation", ""),
            claim_assessments=[
                ClaimAssessment.from_dict(c) for c in data.get("claim_assessments", [])
            ],
            error=data.get("error"),
        )


def _aggregate_bool(values: list[bool]) -> bool | None:
    if not values:
        return None
    if all(values):
        return True
    if not any(values):
        return False
    return None


def build_reliability_diagnostic(
    test_case: TestCase,
    tc_result: TestCaseResult,
    retrieval_results: list[RetrievalEvaluationResult] | None = None,
) -> ReliabilityDiagnostic:
    """Build a consolidated ``ReliabilityDiagnostic`` for one test case.

    ``retrieval_results`` should come from
    ``llm_reliability.rag.evaluators.evaluate_retrieval`` for this test
    case, or be omitted/empty if retrieval was not evaluated. Read-only:
    never re-runs evaluation or retrieval, and never mutates its inputs.
    """
    retrieval_results = retrieval_results or []
    retrieved_chunks = (
        extract_retrieved_chunks(tc_result.model_response)
        if tc_result.model_response is not None
        else None
    )
    retrieved_context = (
        [chunk.text for chunk in retrieved_chunks] if retrieved_chunks is not None else None
    )

    if not tc_result.succeeded:
        return ReliabilityDiagnostic(
            test_case_id=test_case.id,
            question=test_case.input,
            generated_answer=None,
            reference_answer=test_case.reference_answer,
            retrieved_context=None,
            status=DiagnosticStatus.ERROR,
            failure_category=None,
            explanation="Generation failed for this test case; no evaluation evidence exists.",
            error=tc_result.execution_error,
        )

    generated_answer = tc_result.model_response.output_text if tc_result.model_response else None

    correctness_results = [r for r in tc_result.evaluation_results if r.criterion == CORRECTNESS]
    faithfulness_results = [r for r in tc_result.evaluation_results if r.criterion == FAITHFULNESS]

    correctness_judgments = [
        bool(r.passed)
        for r in correctness_results
        if r.status == EvaluationStatus.SUCCESS and r.passed is not None
    ]
    correctness = _aggregate_bool(correctness_judgments)

    faithfulness_labels = {
        r.label
        for r in faithfulness_results
        if r.status == EvaluationStatus.SUCCESS and r.label is not None
    }
    faithfulness = next(iter(faithfulness_labels)) if len(faithfulness_labels) == 1 else None

    hit_scores = [
        r.score > 0.0
        for r in retrieval_results
        if r.status == RetrievalEvaluationStatus.SUCCESS
        and r.metric_name in ("hit_at_k", "mrr")
        and r.score is not None
    ]
    retrieval_hit = _aggregate_bool(hit_scores)

    contributing = sorted(
        {r.evaluator_name for r in correctness_results + faithfulness_results}
        | {r.evaluator_name for r in retrieval_results}
    )

    def _finding(
        status: DiagnosticStatus, category: FailureCategory | None, explanation: str
    ) -> ReliabilityDiagnostic:
        return ReliabilityDiagnostic(
            test_case_id=test_case.id,
            question=test_case.input,
            generated_answer=generated_answer,
            reference_answer=test_case.reference_answer,
            retrieved_context=retrieved_context,
            status=status,
            failure_category=category,
            contributing_evaluators=contributing,
            explanation=explanation,
        )

    if not correctness_results and not faithfulness_results and not retrieval_results:
        return _finding(
            DiagnosticStatus.INSUFFICIENT_EVIDENCE,
            FailureCategory.INSUFFICIENT_EVIDENCE,
            "No evaluator or retrieval evaluator produced usable evidence for this test case.",
        )

    if len(correctness_judgments) >= 2 and correctness is None:
        return _finding(
            DiagnosticStatus.DETERMINED,
            FailureCategory.EVALUATOR_DISAGREEMENT,
            "Correctness evaluators disagreed on whether this answer passed; no single "
            "evaluator's verdict is treated as authoritative.",
        )

    if faithfulness == "contradicted":
        return _finding(
            DiagnosticStatus.DETERMINED,
            FailureCategory.CONTRADICTED,
            "A faithfulness evaluator found the answer contradicted by retrieved context.",
        )

    if faithfulness == "unsupported":
        return _finding(
            DiagnosticStatus.DETERMINED,
            FailureCategory.UNSUPPORTED,
            "A faithfulness evaluator found the answer contains claims the retrieved "
            "context does not address.",
        )

    if retrieval_hit is False and correctness is False:
        return _finding(
            DiagnosticStatus.DETERMINED,
            FailureCategory.RETRIEVAL_FAILURE,
            "Ground-truth-relevant evidence was not retrieved, and the answer was judged "
            "incorrect: consistent with a retrieval failure.",
        )

    if retrieval_hit is True and correctness is False:
        return _finding(
            DiagnosticStatus.DETERMINED,
            FailureCategory.GENERATION_FAILURE,
            "Relevant evidence was retrieved, but the answer was judged incorrect: "
            "consistent with a generation/context-use failure, not a retrieval failure.",
        )

    if correctness is True:
        return _finding(
            DiagnosticStatus.DETERMINED,
            FailureCategory.CORRECT,
            "The answer was judged correct, and no retrieval or faithfulness evidence "
            "contradicts that.",
        )

    if correctness is False:
        return _finding(
            DiagnosticStatus.DETERMINED,
            FailureCategory.INCORRECT,
            "The answer was judged incorrect; no retrieval evidence is available to "
            "attribute the cause further.",
        )

    return _finding(
        DiagnosticStatus.UNDETERMINED,
        FailureCategory.UNDETERMINED,
        "The available evidence does not clearly support any single category.",
    )
