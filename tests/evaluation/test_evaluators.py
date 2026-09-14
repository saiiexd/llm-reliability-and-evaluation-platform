"""Tests for the baseline exact-match evaluator."""

from llm_reliability.evaluation import ExactMatchEvaluator, ModelResponse, TestCase


def _response(text: str) -> ModelResponse:
    return ModelResponse(test_case_id="t1", output_text=text, model_id="mock-adapter-v1")


class TestExactMatchEvaluator:
    def test_exact_match_passes(self):
        test_case = TestCase(id="t1", input="q", reference_answer="Paris")
        result = ExactMatchEvaluator().evaluate(test_case, _response("Paris"))
        assert result.passed is True
        assert result.score == 1.0
        assert result.label == "match"
        assert result.error is None

    def test_match_is_case_and_whitespace_insensitive(self):
        test_case = TestCase(id="t1", input="q", reference_answer="Paris")
        result = ExactMatchEvaluator().evaluate(test_case, _response("  paris  "))
        assert result.passed is True

    def test_mismatch_fails(self):
        test_case = TestCase(id="t1", input="q", reference_answer="Paris")
        result = ExactMatchEvaluator().evaluate(test_case, _response("London"))
        assert result.passed is False
        assert result.score == 0.0
        assert result.label == "mismatch"

    def test_missing_reference_is_not_applicable_not_a_failure(self):
        test_case = TestCase(id="t1", input="q")
        result = ExactMatchEvaluator().evaluate(test_case, _response("anything"))
        assert result.passed is None
        assert result.score is None
        assert result.label == "no_reference"
        assert result.error is None

    def test_result_identifies_evaluator_and_test_case(self):
        test_case = TestCase(id="t42", input="q", reference_answer="x")
        result = ExactMatchEvaluator().evaluate(test_case, _response("x"))
        assert result.evaluator_name == "exact_match"
        assert result.test_case_id == "t42"
