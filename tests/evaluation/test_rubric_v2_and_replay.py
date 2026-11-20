"""Judge rubric v2 (EXP-001 findings E1-E3) and replaying stored answers."""

import json

import pytest

from llm_reliability.api import live
from llm_reliability.evaluation import (
    Dataset,
    ExactMatchEvaluator,
    MockAdapter,
    ModelConfig,
    TestCase,
)
from llm_reliability.evaluation.adapters import ModelAdapter
from llm_reliability.evaluation.llm_judge import (
    CORRECTNESS_RUBRIC,
    CORRECTNESS_RUBRIC_V2,
    LLMJudgeEvaluator,
)
from llm_reliability.evaluation.models import ExecutionRequest, ModelResponse
from llm_reliability.evaluation.replay import ReplayAdapter
from llm_reliability.experiments.local_repository import LocalFileExperimentRepository
from llm_reliability.experiments.manager import ExperimentManager
from llm_reliability.rag.faithfulness_judge import (
    FAITHFULNESS_RUBRIC_V2,
    LLMFaithfulnessEvaluator,
)


class RecordingJudge(ModelAdapter):
    def __init__(self, outcome: str) -> None:
        self.outcome = outcome
        self.prompts: list[str] = []

    def generate(self, request: ExecutionRequest) -> ModelResponse:
        self.prompts.append(request.input_text)
        return ModelResponse(
            test_case_id=request.test_case_id,
            output_text=json.dumps({"outcome": self.outcome, "reasoning": "One sentence."}),
            model_id="judge",
        )


def _case() -> TestCase:
    return TestCase(
        id="q1",
        input="Which wing?",
        reference_answer="Henry Cole wing",
        metadata={"gold_answers": ["Henry Cole wing", "the Henry Cole wing", "Henry Cole wing"]},
    )


def _answer(text: str = "the Henry Cole wing", rag: bool = False) -> ModelResponse:
    meta = {}
    if rag:
        meta["rag"] = {
            "retrieved_chunks": [
                {"chunk_id": "c1", "document_id": "d", "text": "ctx", "score": 1.0, "rank": 1}
            ]
        }
    return ModelResponse(test_case_id="q1", output_text=text, model_id="m", provider_metadata=meta)


def test_v1_correctness_prompt_is_unchanged_and_uses_single_reference():
    judge = RecordingJudge("correct")
    LLMJudgeEvaluator(judge, ModelConfig("judge"), rubric=CORRECTNESS_RUBRIC).evaluate(
        _case(), _answer()
    )
    assert "Reference answer: Henry Cole wing\n" in judge.prompts[0]
    assert CORRECTNESS_RUBRIC.rubric_id == "correctness_vs_reference_v1"


def test_v2_correctness_lists_every_distinct_accepted_answer():
    judge = RecordingJudge("correct")
    result = LLMJudgeEvaluator(judge, ModelConfig("judge"), rubric=CORRECTNESS_RUBRIC_V2).evaluate(
        _case(), _answer()
    )
    assert "Reference answer: Henry Cole wing | the Henry Cole wing\n" in judge.prompts[0]
    assert result.passed is True
    assert result.details["rubric_id"] == "correctness_vs_reference_v2"


def test_v2_faithfulness_no_claim_is_neither_pass_nor_fail():
    judge = RecordingJudge("no_claim")
    result = LLMFaithfulnessEvaluator(
        judge, ModelConfig("judge"), rubric=FAITHFULNESS_RUBRIC_V2
    ).evaluate(_case(), _answer("I don't know", rag=True))
    assert result.label == "no_claim"
    assert result.passed is None
    assert "no_claim" in judge.prompts[0]


def test_v2_faithfulness_supported_still_passes():
    result = LLMFaithfulnessEvaluator(
        RecordingJudge("supported"), ModelConfig("judge"), rubric=FAITHFULNESS_RUBRIC_V2
    ).evaluate(_case(), _answer(rag=True))
    assert result.passed is True


def test_live_judges_rebuild_either_known_rubric_version(monkeypatch):
    monkeypatch.setenv("LLM_RELIABILITY_OPENAI_BASE_URL", "http://localhost:9/v1")
    for name in ("llm_judge", "llm_faithfulness_judge"):
        for version in ("v1", "v2"):
            judge = live.build_live_judge(
                name,
                False,
                judge_model_id="qwen3:4b-instruct",
                for_execution=False,
                rubric_version=version,
            )
            declared = judge.get_config()
            rebuilt = live.build_live_judge(name, True, declared=declared)
            assert rebuilt.get_config() == declared
    with pytest.raises(ValueError, match="Unknown rubric version"):
        live.build_live_judge("llm_judge", True, judge_model_id="m", rubric_version="v9")


def _source_run(tmp_path, responses):
    manager = ExperimentManager(LocalFileExperimentRepository(tmp_path))
    definition = manager.create_experiment(
        experiment_id="src",
        name="src",
        dataset=Dataset(
            name="d",
            test_cases=[
                TestCase(id="q1", input="a?", reference_answer="x"),
                TestCase(id="q2", input="b?", reference_answer="y"),
            ],
        ),
        model_config=ModelConfig(model_id="mock-adapter-v1"),
        evaluators=[ExactMatchEvaluator()],
    )
    run = manager.execute_experiment(
        definition, MockAdapter(responses=responses), [ExactMatchEvaluator()]
    )
    return manager, definition, run


def test_replay_returns_recorded_answers_with_provenance(tmp_path):
    manager, definition, source = _source_run(tmp_path, {"q1": "x", "q2": "wrong"})
    replay = manager.execute_experiment(
        definition, ReplayAdapter(source), [ExactMatchEvaluator()], adapter_id="replay"
    )
    outputs = {
        r.test_case_id: r.model_response.output_text for r in replay.result.test_case_results
    }
    assert outputs == {"q1": "x", "q2": "wrong"}
    meta = replay.result.test_case_results[0].model_response.provider_metadata["replay"]
    assert meta == {"source_run_id": source.run_id, "source_experiment_id": "src"}
    assert replay.result.num_passed == source.result.num_passed


def test_replay_of_a_failed_generation_is_a_failure_not_an_answer(tmp_path):
    class Failing(ModelAdapter):
        def generate(self, request):
            if request.test_case_id == "q2":
                raise RuntimeError("boom")
            return ModelResponse(test_case_id=request.test_case_id, output_text="x", model_id="m")

    manager, definition, _ = _source_run(tmp_path, {"q1": "x", "q2": "y"})
    source = manager.execute_experiment(definition, Failing(), [ExactMatchEvaluator()])
    replay = manager.execute_experiment(definition, ReplayAdapter(source), [ExactMatchEvaluator()])
    by_id = {r.test_case_id: r for r in replay.result.test_case_results}
    assert by_id["q2"].succeeded is False and by_id["q2"].passed is False
    assert by_id["q1"].passed is True
