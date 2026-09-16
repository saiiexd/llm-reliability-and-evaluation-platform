"""
Tests for LLMFaithfulnessEvaluator.

The fake judge is MockAdapter, exactly as for LLMJudgeEvaluator (Prompt 3):
no separate fake-judge class exists.
"""

import json

from llm_reliability.evaluation import (
    EvaluationStatus,
    MockAdapter,
    ModelConfig,
    ModelResponse,
    TestCase,
)
from llm_reliability.evaluation.criteria import FAITHFULNESS
from llm_reliability.rag.faithfulness_judge import FAITHFULNESS_RUBRIC, LLMFaithfulnessEvaluator


def _judge_config() -> ModelConfig:
    return ModelConfig(model_id="fake-judge-v1")


def _rag_response(output_text: str, context_texts: list[str]) -> ModelResponse:
    return ModelResponse(
        test_case_id="t1",
        output_text=output_text,
        model_id="system-under-test",
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


def _plain_response(output_text: str = "answer") -> ModelResponse:
    return ModelResponse(test_case_id="t1", output_text=output_text, model_id="system-under-test")


def _test_case() -> TestCase:
    return TestCase(id="t1", input="What is the capital of France?")


def _judge_json(outcome: str, reasoning: str = "because") -> str:
    return json.dumps({"outcome": outcome, "reasoning": reasoning})


class TestLLMFaithfulnessEvaluator:
    def test_supported_response(self):
        judge = MockAdapter(responses={"t1": _judge_json("supported", "Matches context.")})
        evaluator = LLMFaithfulnessEvaluator(
            judge_adapter=judge, judge_model_config=_judge_config()
        )
        result = evaluator.evaluate(
            _test_case(), _rag_response("Paris.", ["Paris is the capital."])
        )

        assert result.status == EvaluationStatus.SUCCESS
        assert result.label == "supported"
        assert result.passed is True
        assert result.criterion == FAITHFULNESS
        assert result.score is None

    def test_unsupported_response(self):
        judge = MockAdapter(responses={"t1": _judge_json("unsupported")})
        evaluator = LLMFaithfulnessEvaluator(
            judge_adapter=judge, judge_model_config=_judge_config()
        )
        result = evaluator.evaluate(
            _test_case(), _rag_response("The population is 2 million.", ["Paris is the capital."])
        )
        assert result.label == "unsupported"
        assert result.passed is False

    def test_contradicted_response(self):
        judge = MockAdapter(responses={"t1": _judge_json("contradicted")})
        evaluator = LLMFaithfulnessEvaluator(
            judge_adapter=judge, judge_model_config=_judge_config()
        )
        result = evaluator.evaluate(
            _test_case(), _rag_response("Lyon is the capital.", ["Paris is the capital."])
        )
        assert result.label == "contradicted"
        assert result.passed is False

    def test_undetermined_response_is_neither_passed_nor_failed(self):
        judge = MockAdapter(responses={"t1": _judge_json("undetermined")})
        evaluator = LLMFaithfulnessEvaluator(
            judge_adapter=judge, judge_model_config=_judge_config()
        )
        result = evaluator.evaluate(
            _test_case(), _rag_response("Ambiguous answer.", ["Some unrelated context."])
        )
        assert result.label == "undetermined"
        assert result.passed is None
        assert result.status == EvaluationStatus.SUCCESS

    def test_missing_context_is_skipped(self):
        judge = MockAdapter(responses={"t1": _judge_json("supported")})
        evaluator = LLMFaithfulnessEvaluator(
            judge_adapter=judge, judge_model_config=_judge_config()
        )
        result = evaluator.evaluate(_test_case(), _plain_response())
        assert result.status == EvaluationStatus.SKIPPED
        assert result.label == "no_context"

    def test_empty_answer_is_skipped(self):
        judge = MockAdapter(responses={"t1": _judge_json("supported")})
        evaluator = LLMFaithfulnessEvaluator(
            judge_adapter=judge, judge_model_config=_judge_config()
        )
        result = evaluator.evaluate(_test_case(), _rag_response("", ["Paris is the capital."]))
        assert result.status == EvaluationStatus.SKIPPED
        assert result.label == "empty_answer"

    def test_whitespace_only_answer_is_skipped(self):
        judge = MockAdapter(responses={"t1": _judge_json("supported")})
        evaluator = LLMFaithfulnessEvaluator(
            judge_adapter=judge, judge_model_config=_judge_config()
        )
        result = evaluator.evaluate(_test_case(), _rag_response("   ", ["Paris is the capital."]))
        assert result.status == EvaluationStatus.SKIPPED

    def test_malformed_judge_output_is_output_validation_error(self):
        judge = MockAdapter(responses={"t1": "I think it looks fine."})
        evaluator = LLMFaithfulnessEvaluator(
            judge_adapter=judge, judge_model_config=_judge_config()
        )
        result = evaluator.evaluate(
            _test_case(), _rag_response("Paris.", ["Paris is the capital."])
        )
        assert result.status == EvaluationStatus.OUTPUT_VALIDATION_ERROR
        assert result.error is not None

    def test_judge_provider_failure_is_execution_error(self):
        judge = MockAdapter(errors={"t1": "provider outage"})
        evaluator = LLMFaithfulnessEvaluator(
            judge_adapter=judge, judge_model_config=_judge_config()
        )
        result = evaluator.evaluate(
            _test_case(), _rag_response("Paris.", ["Paris is the capital."])
        )
        assert result.status == EvaluationStatus.EXECUTION_ERROR
        assert "provider outage" in result.error

    def test_result_carries_judge_model_and_rubric_config(self):
        judge = MockAdapter(responses={"t1": _judge_json("supported")})
        evaluator = LLMFaithfulnessEvaluator(
            judge_adapter=judge, judge_model_config=_judge_config()
        )
        result = evaluator.evaluate(
            _test_case(), _rag_response("Paris.", ["Paris is the capital."])
        )
        assert result.evaluator_config["judge_model_id"] == "fake-judge-v1"
        assert result.evaluator_config["rubric"]["rubric_id"] == FAITHFULNESS_RUBRIC.rubric_id

    def test_default_rubric_has_four_way_scale(self):
        assert FAITHFULNESS_RUBRIC.scale == (
            "contradicted",
            "unsupported",
            "undetermined",
            "supported",
        )
