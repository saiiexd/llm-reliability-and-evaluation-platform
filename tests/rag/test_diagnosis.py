"""Tests for the evidence-based RAG failure diagnosis decision table."""

from llm_reliability.evaluation import EvaluationResult, EvaluationStatus
from llm_reliability.evaluation.criteria import CORRECTNESS
from llm_reliability.rag.diagnosis import RagDiagnosisCategory, diagnose_rag_execution
from llm_reliability.rag.evaluators import RetrievalEvaluationResult, RetrievalEvaluationStatus


def _hit(score: float, metric_name: str = "hit_at_k") -> RetrievalEvaluationResult:
    return RetrievalEvaluationResult(
        evaluator_name="hit_at_3",
        test_case_id="t1",
        metric_name=metric_name,
        status=RetrievalEvaluationStatus.SUCCESS,
        score=score,
    )


def _skipped_retrieval() -> RetrievalEvaluationResult:
    return RetrievalEvaluationResult(
        evaluator_name="hit_at_3",
        test_case_id="t1",
        metric_name="hit_at_k",
        status=RetrievalEvaluationStatus.SKIPPED,
    )


def _answer(passed: bool) -> EvaluationResult:
    return EvaluationResult(
        evaluator_name="exact_match",
        test_case_id="t1",
        status=EvaluationStatus.SUCCESS,
        criterion=CORRECTNESS,
        passed=passed,
    )


def _skipped_answer() -> EvaluationResult:
    return EvaluationResult(
        evaluator_name="exact_match",
        test_case_id="t1",
        status=EvaluationStatus.SKIPPED,
        criterion=CORRECTNESS,
    )


class TestDiagnoseRagExecution:
    def test_retrieval_evidence_unavailable_when_no_ground_truth(self):
        diagnosis = diagnose_rag_execution("t1", [], [_answer(True)])
        assert diagnosis.category == RagDiagnosisCategory.RETRIEVAL_EVIDENCE_UNAVAILABLE

    def test_retrieval_evidence_unavailable_when_retrieval_skipped(self):
        diagnosis = diagnose_rag_execution("t1", [_skipped_retrieval()], [_answer(False)])
        assert diagnosis.category == RagDiagnosisCategory.RETRIEVAL_EVIDENCE_UNAVAILABLE

    def test_successful_grounded_execution(self):
        diagnosis = diagnose_rag_execution("t1", [_hit(1.0)], [_answer(True)])
        assert diagnosis.category == RagDiagnosisCategory.SUCCESSFUL_GROUNDED_EXECUTION

    def test_generation_failure_despite_relevant_context(self):
        diagnosis = diagnose_rag_execution("t1", [_hit(1.0)], [_answer(False)])
        assert (
            diagnosis.category == RagDiagnosisCategory.GENERATION_FAILURE_DESPITE_RELEVANT_CONTEXT
        )

    def test_retrieval_failure_when_missed_and_answer_wrong(self):
        diagnosis = diagnose_rag_execution("t1", [_hit(0.0)], [_answer(False)])
        assert diagnosis.category == RagDiagnosisCategory.RETRIEVAL_FAILURE

    def test_undetermined_when_missed_but_answer_correct(self):
        diagnosis = diagnose_rag_execution("t1", [_hit(0.0)], [_answer(True)])
        assert diagnosis.category == RagDiagnosisCategory.UNDETERMINED

    def test_undetermined_when_retrieval_hit_but_no_answer_evidence(self):
        diagnosis = diagnose_rag_execution("t1", [_hit(1.0)], [_skipped_answer()])
        assert diagnosis.category == RagDiagnosisCategory.UNDETERMINED

    def test_incorrect_answer_alone_is_never_retrieval_failure_without_retrieval_evidence(self):
        diagnosis = diagnose_rag_execution("t1", [], [_answer(False)])
        assert diagnosis.category != RagDiagnosisCategory.RETRIEVAL_FAILURE
        assert diagnosis.category == RagDiagnosisCategory.RETRIEVAL_EVIDENCE_UNAVAILABLE

    def test_evidence_is_preserved_on_the_diagnosis(self):
        diagnosis = diagnose_rag_execution("t1", [_hit(1.0)], [_answer(True)])
        assert diagnosis.evidence == {"retrieval_hit": True, "answer_correct": True}

    def test_disagreeing_retrieval_signals_are_treated_as_unavailable(self):
        mixed = [_hit(1.0), _hit(0.0, metric_name="mrr")]
        diagnosis = diagnose_rag_execution("t1", mixed, [_answer(True)])
        # Disagreement among retrieval signals aggregates to None (unavailable),
        # which must never be silently resolved into a confident category.
        assert diagnosis.evidence["retrieval_hit"] is None
        assert diagnosis.category == RagDiagnosisCategory.RETRIEVAL_EVIDENCE_UNAVAILABLE

    def test_round_trip_through_dict(self):
        diagnosis = diagnose_rag_execution("t1", [_hit(1.0)], [_answer(True)])
        from llm_reliability.rag.diagnosis import RagDiagnosis

        reloaded = RagDiagnosis.from_dict(diagnosis.to_dict())
        assert reloaded == diagnosis

    def test_never_uses_the_word_hallucination(self):
        for category in RagDiagnosisCategory:
            assert "hallucination" not in category.value
