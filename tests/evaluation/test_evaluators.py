"""Tests for the baseline exact-match evaluator."""

from llm_reliability.evaluation import (
    EvaluationStatus,
    ExactMatchEvaluator,
    ModelResponse,
    TestCase,
)
from llm_reliability.evaluation.criteria import CORRECTNESS


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
        assert result.status == EvaluationStatus.SUCCESS
        assert result.criterion == CORRECTNESS

    def test_match_is_case_and_whitespace_insensitive_by_default(self):
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
        assert result.status == EvaluationStatus.SKIPPED

    def test_result_identifies_evaluator_and_test_case(self):
        test_case = TestCase(id="t42", input="q", reference_answer="x")
        result = ExactMatchEvaluator().evaluate(test_case, _response("x"))
        assert result.evaluator_name == "exact_match"
        assert result.test_case_id == "t42"

    def test_result_carries_its_own_configuration(self):
        evaluator = ExactMatchEvaluator(case_sensitive=True, collapse_whitespace=False)
        test_case = TestCase(id="t1", input="q", reference_answer="Paris")
        result = evaluator.evaluate(test_case, _response("Paris"))
        assert result.evaluator_config == {"case_sensitive": True, "collapse_whitespace": False}


class TestExactMatchNormalization:
    def test_default_collapses_internal_whitespace(self):
        test_case = TestCase(id="t1", input="q", reference_answer="Paris is nice")
        result = ExactMatchEvaluator().evaluate(test_case, _response("Paris   is\tnice"))
        assert result.passed is True

    def test_case_sensitive_true_rejects_case_difference(self):
        evaluator = ExactMatchEvaluator(case_sensitive=True)
        test_case = TestCase(id="t1", input="q", reference_answer="Paris")
        result = evaluator.evaluate(test_case, _response("paris"))
        assert result.passed is False

    def test_case_sensitive_true_accepts_exact_case_match(self):
        evaluator = ExactMatchEvaluator(case_sensitive=True)
        test_case = TestCase(id="t1", input="q", reference_answer="Paris")
        result = evaluator.evaluate(test_case, _response("Paris"))
        assert result.passed is True

    def test_collapse_whitespace_false_requires_identical_whitespace(self):
        evaluator = ExactMatchEvaluator(collapse_whitespace=False)
        test_case = TestCase(id="t1", input="q", reference_answer="Paris is nice")
        result = evaluator.evaluate(test_case, _response("Paris  is nice"))
        assert result.passed is False

    def test_collapse_whitespace_false_still_matches_identical_text(self):
        evaluator = ExactMatchEvaluator(collapse_whitespace=False)
        test_case = TestCase(id="t1", input="q", reference_answer="Paris is nice")
        result = evaluator.evaluate(test_case, _response("Paris is nice"))
        assert result.passed is True

    def test_normalization_does_not_strip_punctuation(self):
        test_case = TestCase(id="t1", input="q", reference_answer="Paris.")
        result = ExactMatchEvaluator().evaluate(test_case, _response("Paris"))
        assert result.passed is False
