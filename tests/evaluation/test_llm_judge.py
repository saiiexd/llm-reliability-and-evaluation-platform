"""
Tests for the LLM-as-a-judge evaluator, rubric, and output parsing.

The fake judge is simply MockAdapter (from the Core Evaluation Engine)
used as the judge adapter: no separate fake-judge class exists, since a
judge model's fundamental capability -- receive text, produce text -- is
identical to any other ModelAdapter's. Unit tests here never require a
real external judge model or API credentials.
"""

import json

import pytest

from llm_reliability.evaluation import (
    EvaluationStatus,
    MockAdapter,
    ModelConfig,
    ModelResponse,
    TestCase,
)
from llm_reliability.evaluation.criteria import CORRECTNESS
from llm_reliability.evaluation.llm_judge import (
    CORRECTNESS_RUBRIC,
    JudgeOutputValidationError,
    JudgeRubric,
    LLMJudgeEvaluator,
    parse_judge_output,
    render_judge_prompt,
)


def _judge_config() -> ModelConfig:
    return ModelConfig(model_id="fake-judge-v1")


def _test_case(reference_answer: str | None = "4") -> TestCase:
    return TestCase(id="t1", input="What is 2+2?", reference_answer=reference_answer)


def _response(text: str = "4") -> ModelResponse:
    return ModelResponse(test_case_id="t1", output_text=text, model_id="system-under-test")


def _judge_response(outcome: str, reasoning: str = "Because it matches.") -> str:
    return json.dumps({"outcome": outcome, "reasoning": reasoning})


class TestJudgeRubric:
    def test_requires_non_empty_rubric_id(self):
        with pytest.raises(ValueError, match="rubric_id"):
            JudgeRubric(rubric_id="", criterion=CORRECTNESS, scale=("a", "b"), instructions="x")

    def test_requires_at_least_two_scale_values(self):
        with pytest.raises(ValueError, match="at least two"):
            JudgeRubric(
                rubric_id="r1", criterion=CORRECTNESS, scale=("only_one",), instructions="x"
            )

    def test_requires_unique_scale_values(self):
        with pytest.raises(ValueError, match="unique"):
            JudgeRubric(rubric_id="r1", criterion=CORRECTNESS, scale=("a", "a"), instructions="x")

    def test_requires_non_empty_instructions(self):
        with pytest.raises(ValueError, match="instructions"):
            JudgeRubric(rubric_id="r1", criterion=CORRECTNESS, scale=("a", "b"), instructions="")

    def test_round_trip_through_dict(self):
        rubric = JudgeRubric(
            rubric_id="r1",
            criterion=CORRECTNESS,
            scale=("incorrect", "correct"),
            instructions="Judge it.",
        )
        reloaded = JudgeRubric.from_dict(rubric.to_dict())
        assert reloaded == rubric

    def test_default_rubric_has_three_way_scale(self):
        assert CORRECTNESS_RUBRIC.scale == ("incorrect", "partially_correct", "correct")


class TestRenderJudgePrompt:
    def test_prompt_includes_question_reference_and_candidate(self):
        prompt = render_judge_prompt(
            CORRECTNESS_RUBRIC,
            question="What is 2+2?",
            reference_answer="4",
            candidate_answer="four",
        )
        assert "What is 2+2?" in prompt
        assert "4" in prompt
        assert "four" in prompt
        assert CORRECTNESS_RUBRIC.instructions in prompt


class TestParseJudgeOutput:
    def test_valid_output_parses(self):
        parsed = parse_judge_output(
            _judge_response("correct", "Matches exactly."), CORRECTNESS_RUBRIC
        )
        assert parsed.outcome == "correct"
        assert parsed.reasoning == "Matches exactly."

    def test_invalid_json_raises(self):
        with pytest.raises(JudgeOutputValidationError, match="not valid JSON"):
            parse_judge_output("this is not json", CORRECTNESS_RUBRIC)

    def test_non_object_json_raises(self):
        with pytest.raises(JudgeOutputValidationError, match="object"):
            parse_judge_output("[1, 2, 3]", CORRECTNESS_RUBRIC)

    def test_outcome_outside_scale_raises(self):
        with pytest.raises(JudgeOutputValidationError, match="outcome"):
            parse_judge_output(_judge_response("definitely_correct"), CORRECTNESS_RUBRIC)

    def test_missing_reasoning_raises(self):
        with pytest.raises(JudgeOutputValidationError, match="reasoning"):
            parse_judge_output(json.dumps({"outcome": "correct"}), CORRECTNESS_RUBRIC)

    def test_empty_reasoning_raises(self):
        with pytest.raises(JudgeOutputValidationError, match="reasoning"):
            parse_judge_output(_judge_response("correct", reasoning=""), CORRECTNESS_RUBRIC)


class TestLLMJudgeEvaluator:
    def test_valid_correct_response(self):
        judge = MockAdapter(
            responses={"t1": _judge_response("correct", "It matches the reference.")}
        )
        evaluator = LLMJudgeEvaluator(judge_adapter=judge, judge_model_config=_judge_config())
        result = evaluator.evaluate(_test_case(), _response())

        assert result.status == EvaluationStatus.SUCCESS
        assert result.label == "correct"
        assert result.passed is True
        assert result.explanation == "It matches the reference."
        assert result.criterion == CORRECTNESS
        assert result.score is None  # categorical judgment, not a numeric score

    def test_valid_partially_correct_response_is_not_passed(self):
        judge = MockAdapter(responses={"t1": _judge_response("partially_correct")})
        evaluator = LLMJudgeEvaluator(judge_adapter=judge, judge_model_config=_judge_config())
        result = evaluator.evaluate(_test_case(), _response())
        assert result.label == "partially_correct"
        assert result.passed is False

    def test_valid_incorrect_response(self):
        judge = MockAdapter(responses={"t1": _judge_response("incorrect")})
        evaluator = LLMJudgeEvaluator(judge_adapter=judge, judge_model_config=_judge_config())
        result = evaluator.evaluate(_test_case(), _response())
        assert result.label == "incorrect"
        assert result.passed is False

    def test_borderline_response_with_minimal_reasoning(self):
        judge = MockAdapter(
            responses={"t1": _judge_response("partially_correct", "Close but incomplete.")}
        )
        evaluator = LLMJudgeEvaluator(judge_adapter=judge, judge_model_config=_judge_config())
        result = evaluator.evaluate(_test_case(), _response())
        assert result.status == EvaluationStatus.SUCCESS
        assert result.label == "partially_correct"

    def test_invalid_judge_response_produces_output_validation_error(self):
        judge = MockAdapter(responses={"t1": "The answer looks correct to me."})
        evaluator = LLMJudgeEvaluator(judge_adapter=judge, judge_model_config=_judge_config())
        result = evaluator.evaluate(_test_case(), _response())

        assert result.status == EvaluationStatus.OUTPUT_VALIDATION_ERROR
        assert result.error is not None
        assert result.passed is None
        assert result.label is None
        assert result.details["raw_judge_output"] == "The answer looks correct to me."

    def test_judge_adapter_failure_produces_execution_error(self):
        judge = MockAdapter(errors={"t1": "judge API unavailable"})
        evaluator = LLMJudgeEvaluator(judge_adapter=judge, judge_model_config=_judge_config())
        result = evaluator.evaluate(_test_case(), _response())

        assert result.status == EvaluationStatus.EXECUTION_ERROR
        assert "judge API unavailable" in result.error

    def test_missing_reference_is_skipped_without_calling_judge(self):
        judge = MockAdapter(responses={"t1": _judge_response("correct")})
        evaluator = LLMJudgeEvaluator(judge_adapter=judge, judge_model_config=_judge_config())
        result = evaluator.evaluate(_test_case(reference_answer=None), _response())

        assert result.status == EvaluationStatus.SKIPPED
        assert result.label == "no_reference"

    def test_result_carries_judge_model_id_and_rubric_in_config(self):
        judge = MockAdapter(responses={"t1": _judge_response("correct")})
        evaluator = LLMJudgeEvaluator(judge_adapter=judge, judge_model_config=_judge_config())
        result = evaluator.evaluate(_test_case(), _response())

        assert result.evaluator_config["judge_model_id"] == "fake-judge-v1"
        assert result.evaluator_config["rubric"]["rubric_id"] == CORRECTNESS_RUBRIC.rubric_id

    def test_custom_rubric_is_used_for_validation(self):
        rubric = JudgeRubric(
            rubric_id="binary_v1",
            criterion=CORRECTNESS,
            scale=("wrong", "right"),
            instructions="Answer 'wrong' or 'right'.",
        )
        judge = MockAdapter(responses={"t1": json.dumps({"outcome": "right", "reasoning": "ok"})})
        evaluator = LLMJudgeEvaluator(
            judge_adapter=judge, judge_model_config=_judge_config(), rubric=rubric
        )
        result = evaluator.evaluate(_test_case(), _response())

        assert result.label == "right"
        assert result.passed is True  # "right" is the last (best) scale entry

    def test_judge_is_called_at_most_once_per_test_case(self):
        call_count = {"n": 0}

        class CountingAdapter(MockAdapter):
            def generate(self, request):
                call_count["n"] += 1
                return super().generate(request)

        judge = CountingAdapter(responses={"t1": _judge_response("correct")})
        evaluator = LLMJudgeEvaluator(judge_adapter=judge, judge_model_config=_judge_config())
        evaluator.evaluate(_test_case(), _response())

        assert call_count["n"] == 1
