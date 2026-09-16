"""Tests for RunResult serialization determinism and machine-readability."""

import json

from llm_reliability.evaluation import (
    Dataset,
    EvaluationRunner,
    ExactMatchEvaluator,
    MockAdapter,
    ModelConfig,
    TestCase,
)


def _run() -> EvaluationRunner:
    return EvaluationRunner(
        adapter=MockAdapter(responses={"t1": "4"}), evaluators=[ExactMatchEvaluator()]
    )


def _dataset() -> Dataset:
    return Dataset(
        name="serialization-test",
        test_cases=[TestCase(id="t1", input="2+2?", reference_answer="4")],
    )


class TestRunResultSerialization:
    def test_to_dict_returns_plain_python_types(self):
        run_result = _run().run(_dataset(), ModelConfig(model_id="mock-adapter-v1"))
        as_dict = run_result.to_dict()
        assert isinstance(as_dict, dict)
        assert as_dict["dataset_name"] == "serialization-test"
        assert as_dict["test_case_results"][0]["test_case_id"] == "t1"

    def test_to_json_is_valid_and_round_trips(self):
        run_result = _run().run(_dataset(), ModelConfig(model_id="mock-adapter-v1"))
        json_text = run_result.to_json()
        parsed = json.loads(json_text)
        assert parsed["model_id"] == "mock-adapter-v1"
        assert parsed["test_case_results"][0]["model_response"]["output_text"] == "4"

    def test_to_json_is_deterministic_across_identical_runs(self):
        dataset = _dataset()
        config = ModelConfig(model_id="mock-adapter-v1")
        first = _run().run(dataset, config).to_json()
        second = _run().run(dataset, config).to_json()
        assert first == second


class TestRunResultDeserialization:
    def test_from_json_round_trips_a_successful_run(self):
        run_result = _run().run(_dataset(), ModelConfig(model_id="mock-adapter-v1"))
        reloaded = run_result.__class__.from_json(run_result.to_json())
        assert reloaded == run_result

    def test_from_dict_round_trips_a_failed_test_case(self):
        adapter = MockAdapter(errors={"t1": "boom"})
        run_result = EvaluationRunner(adapter=adapter, evaluators=[ExactMatchEvaluator()]).run(
            _dataset(), ModelConfig(model_id="mock-adapter-v1")
        )
        reloaded = run_result.__class__.from_dict(run_result.to_dict())
        assert reloaded == run_result
        assert reloaded.test_case_results[0].model_response is None
        assert reloaded.test_case_results[0].execution_error is not None

    def test_from_dict_round_trips_dataset_test_case_and_model_config(self):
        dataset = _dataset()
        reloaded_dataset = dataset.__class__.from_dict(
            {"name": dataset.name, "test_cases": [{"id": "t1", "input": "2+2?"}]}
        )
        assert reloaded_dataset.test_cases[0].id == "t1"

        config = ModelConfig(model_id="mock-adapter-v1", temperature=0.5)
        reloaded_config = ModelConfig.from_dict({"model_id": "mock-adapter-v1", "temperature": 0.5})
        assert reloaded_config == config
