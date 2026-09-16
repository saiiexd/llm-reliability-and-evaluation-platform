"""Tests for the ContextSupportBaseline evaluator and its overlap computation."""

import pytest

from llm_reliability.evaluation import EvaluationStatus, ModelResponse, TestCase
from llm_reliability.evaluation.criteria import CONTEXT_SUPPORT
from llm_reliability.rag.context_support import ContextSupportBaseline, compute_overlap_ratio


def _rag_response(output_text: str, context_texts: list[str]) -> ModelResponse:
    return ModelResponse(
        test_case_id="t1",
        output_text=output_text,
        model_id="m1",
        provider_metadata={
            "rag": {
                "query": "q",
                "retrieved_chunks": [
                    {
                        "chunk_id": f"c{i}",
                        "document_id": "d1",
                        "text": text,
                        "score": 1.0,
                        "rank": i + 1,
                    }
                    for i, text in enumerate(context_texts)
                ],
                "retrieval_config": {},
                "pipeline_config": {},
                "prompt": "prompt",
            }
        },
    )


def _plain_response() -> ModelResponse:
    return ModelResponse(test_case_id="t1", output_text="answer", model_id="m1")


class TestComputeOverlapRatio:
    def test_full_overlap_scores_one(self):
        assert compute_overlap_ratio("paris france", "paris is in france") == pytest.approx(1.0)

    def test_no_overlap_scores_zero(self):
        assert compute_overlap_ratio("bananas apples", "paris france") == 0.0

    def test_partial_overlap_is_fractional(self):
        assert compute_overlap_ratio("paris bananas", "paris is nice") == pytest.approx(0.5)

    def test_empty_answer_scores_zero(self):
        assert compute_overlap_ratio("", "some context") == 0.0

    def test_case_insensitive_by_default(self):
        assert compute_overlap_ratio("PARIS", "paris is a city") == pytest.approx(1.0)

    def test_case_sensitive_when_configured(self):
        assert compute_overlap_ratio("PARIS", "paris is a city", case_sensitive=True) == 0.0


class TestContextSupportBaseline:
    def test_supported_answer_passes_threshold(self):
        evaluator = ContextSupportBaseline(min_overlap_ratio=0.5)
        tc = TestCase(id="t1", input="q")
        response = _rag_response("Paris is the capital", ["Paris is the capital of France"])
        result = evaluator.evaluate(tc, response)
        assert result.status == EvaluationStatus.SUCCESS
        assert result.passed is True
        assert result.label == "context_supported"
        assert result.criterion == CONTEXT_SUPPORT

    def test_unsupported_answer_fails_threshold(self):
        evaluator = ContextSupportBaseline(min_overlap_ratio=0.9)
        tc = TestCase(id="t1", input="q")
        response = _rag_response(
            "completely different words here", ["Paris is the capital of France"]
        )
        result = evaluator.evaluate(tc, response)
        assert result.passed is False
        assert result.label == "context_unsupported"

    def test_missing_retrieval_evidence_is_skipped(self):
        evaluator = ContextSupportBaseline()
        result = evaluator.evaluate(TestCase(id="t1", input="q"), _plain_response())
        assert result.status == EvaluationStatus.SKIPPED

    def test_invalid_min_overlap_ratio_is_rejected(self):
        with pytest.raises(ValueError, match="min_overlap_ratio"):
            ContextSupportBaseline(min_overlap_ratio=1.5)

    def test_result_carries_its_own_configuration(self):
        evaluator = ContextSupportBaseline(min_overlap_ratio=0.4, case_sensitive=True)
        tc = TestCase(id="t1", input="q")
        response = _rag_response("answer text", ["some context"])
        result = evaluator.evaluate(tc, response)
        assert result.evaluator_config == {"min_overlap_ratio": 0.4, "case_sensitive": True}

    def test_does_not_claim_faithfulness_in_explanation(self):
        evaluator = ContextSupportBaseline()
        tc = TestCase(id="t1", input="q")
        response = _rag_response("answer", ["some context"])
        result = evaluator.evaluate(tc, response)
        assert "not a faithfulness" in result.explanation.lower()
