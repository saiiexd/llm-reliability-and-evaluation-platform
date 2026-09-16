"""
Tests for the consolidated reliability diagnosis decision table.

Covers every precedence step documented in
llm_reliability.diagnostics.diagnosis, plus malformed/missing-evidence
edge cases: generation failure, insufficient evidence, evaluator
disagreement, retrieval failure, generation failure despite relevant
context, contradiction, unsupported, correct, incorrect, and undetermined.
"""

from llm_reliability.diagnostics.diagnosis import build_reliability_diagnostic
from llm_reliability.diagnostics.taxonomy import DiagnosticStatus, FailureCategory
from llm_reliability.evaluation.criteria import CORRECTNESS, FAITHFULNESS
from llm_reliability.evaluation.models import (
    EvaluationResult,
    EvaluationStatus,
    ModelResponse,
    TestCase,
    TestCaseResult,
)
from llm_reliability.rag.evaluators import RetrievalEvaluationResult, RetrievalEvaluationStatus


def _test_case(relevant_chunk_ids=None) -> TestCase:
    metadata = {"relevant_chunk_ids": relevant_chunk_ids} if relevant_chunk_ids is not None else {}
    return TestCase(
        id="t1", input="What is the capital of France?", reference_answer="Paris", metadata=metadata
    )


def _response(output_text: str = "Paris", with_rag: bool = False) -> ModelResponse:
    provider_metadata = {}
    if with_rag:
        provider_metadata["rag"] = {
            "query": "q",
            "retrieved_chunks": [
                {
                    "chunk_id": "c1",
                    "document_id": "d1",
                    "text": "Paris is the capital.",
                    "score": 1.0,
                    "rank": 1,
                }
            ],
            "retrieval_config": {},
            "pipeline_config": {},
            "prompt": "prompt",
        }
    return ModelResponse(
        test_case_id="t1",
        output_text=output_text,
        model_id="m1",
        provider_metadata=provider_metadata,
    )


def _tc_result(evaluation_results, execution_error=None, response=None) -> TestCaseResult:
    return TestCaseResult(
        test_case_id="t1",
        input_text="What is the capital of France?",
        reference_answer="Paris",
        model_response=None if execution_error else (response or _response()),
        evaluation_results=evaluation_results,
        execution_error=execution_error,
    )


def _correctness(passed: bool, evaluator_name: str = "exact_match") -> EvaluationResult:
    return EvaluationResult(
        evaluator_name=evaluator_name,
        test_case_id="t1",
        status=EvaluationStatus.SUCCESS,
        criterion=CORRECTNESS,
        passed=passed,
    )


def _faithfulness(label: str) -> EvaluationResult:
    return EvaluationResult(
        evaluator_name="llm_faithfulness_judge",
        test_case_id="t1",
        status=EvaluationStatus.SUCCESS,
        criterion=FAITHFULNESS,
        label=label,
        passed=(label == "supported"),
    )


def _hit(score: float) -> RetrievalEvaluationResult:
    return RetrievalEvaluationResult(
        evaluator_name="hit_at_1",
        test_case_id="t1",
        metric_name="hit_at_k",
        status=RetrievalEvaluationStatus.SUCCESS,
        score=score,
    )


class TestBuildReliabilityDiagnostic:
    def test_generation_failure_is_error_status(self):
        tc_result = _tc_result([], execution_error="AdapterError: boom")
        diagnosis = build_reliability_diagnostic(_test_case(), tc_result, [])
        assert diagnosis.status == DiagnosticStatus.ERROR
        assert diagnosis.failure_category is None
        assert diagnosis.error == "AdapterError: boom"

    def test_no_evidence_at_all_is_insufficient_evidence(self):
        tc_result = _tc_result([])
        diagnosis = build_reliability_diagnostic(_test_case(), tc_result, [])
        assert diagnosis.status == DiagnosticStatus.INSUFFICIENT_EVIDENCE
        assert diagnosis.failure_category == FailureCategory.INSUFFICIENT_EVIDENCE

    def test_correctness_evaluator_disagreement(self):
        tc_result = _tc_result(
            [_correctness(True, "exact_match"), _correctness(False, "semantic_similarity")]
        )
        diagnosis = build_reliability_diagnostic(_test_case(), tc_result, [])
        assert diagnosis.failure_category == FailureCategory.EVALUATOR_DISAGREEMENT
        assert diagnosis.status == DiagnosticStatus.DETERMINED
        assert set(diagnosis.contributing_evaluators) == {"exact_match", "semantic_similarity"}

    def test_retrieval_failure(self):
        tc_result = _tc_result([_correctness(False)], response=_response("wrong", with_rag=True))
        diagnosis = build_reliability_diagnostic(
            _test_case(relevant_chunk_ids=["c1"]), tc_result, [_hit(0.0)]
        )
        assert diagnosis.failure_category == FailureCategory.RETRIEVAL_FAILURE

    def test_generation_failure_despite_relevant_context(self):
        tc_result = _tc_result([_correctness(False)], response=_response("wrong", with_rag=True))
        diagnosis = build_reliability_diagnostic(
            _test_case(relevant_chunk_ids=["c1"]), tc_result, [_hit(1.0)]
        )
        assert diagnosis.failure_category == FailureCategory.GENERATION_FAILURE

    def test_contradicted_takes_precedence_over_plain_incorrect(self):
        tc_result = _tc_result([_correctness(False), _faithfulness("contradicted")])
        diagnosis = build_reliability_diagnostic(_test_case(), tc_result, [])
        assert diagnosis.failure_category == FailureCategory.CONTRADICTED

    def test_unsupported(self):
        tc_result = _tc_result([_faithfulness("unsupported")])
        diagnosis = build_reliability_diagnostic(_test_case(), tc_result, [])
        assert diagnosis.failure_category == FailureCategory.UNSUPPORTED

    def test_correct(self):
        tc_result = _tc_result([_correctness(True)])
        diagnosis = build_reliability_diagnostic(_test_case(), tc_result, [])
        assert diagnosis.failure_category == FailureCategory.CORRECT
        assert diagnosis.status == DiagnosticStatus.DETERMINED

    def test_incorrect_without_retrieval_evidence(self):
        tc_result = _tc_result([_correctness(False)])
        diagnosis = build_reliability_diagnostic(_test_case(), tc_result, [])
        assert diagnosis.failure_category == FailureCategory.INCORRECT

    def test_undetermined_when_only_skipped_faithfulness_and_no_correctness(self):
        skipped = EvaluationResult(
            evaluator_name="llm_faithfulness_judge",
            test_case_id="t1",
            status=EvaluationStatus.SKIPPED,
            criterion=FAITHFULNESS,
            label="no_context",
        )
        tc_result = _tc_result([skipped])
        diagnosis = build_reliability_diagnostic(_test_case(), tc_result, [])
        assert diagnosis.status == DiagnosticStatus.UNDETERMINED
        assert diagnosis.failure_category == FailureCategory.UNDETERMINED

    def test_question_and_answer_are_preserved(self):
        tc_result = _tc_result([_correctness(True)])
        diagnosis = build_reliability_diagnostic(_test_case(), tc_result, [])
        assert diagnosis.question == "What is the capital of France?"
        assert diagnosis.generated_answer == "Paris"
        assert diagnosis.reference_answer == "Paris"

    def test_retrieved_context_is_extracted_when_present(self):
        tc_result = _tc_result([_correctness(True)], response=_response("Paris", with_rag=True))
        diagnosis = build_reliability_diagnostic(_test_case(), tc_result, [])
        assert diagnosis.retrieved_context == ["Paris is the capital."]

    def test_retrieved_context_is_none_when_not_a_rag_execution(self):
        tc_result = _tc_result([_correctness(True)], response=_response("Paris", with_rag=False))
        diagnosis = build_reliability_diagnostic(_test_case(), tc_result, [])
        assert diagnosis.retrieved_context is None

    def test_confidence_is_not_fabricated(self):
        tc_result = _tc_result([_correctness(True)])
        diagnosis = build_reliability_diagnostic(_test_case(), tc_result, [])
        assert diagnosis.confidence is None

    def test_round_trip_through_dict(self):
        from llm_reliability.diagnostics.diagnosis import ReliabilityDiagnostic

        tc_result = _tc_result([_correctness(True)])
        diagnosis = build_reliability_diagnostic(_test_case(), tc_result, [])
        reloaded = ReliabilityDiagnostic.from_dict(diagnosis.to_dict())
        assert reloaded == diagnosis

    def test_never_uses_the_word_hallucination(self):
        for category in FailureCategory:
            assert "hallucination" not in category.value
