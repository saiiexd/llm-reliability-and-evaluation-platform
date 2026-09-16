"""Tests for RunResult serialization determinism and machine-readability."""

import json

from llm_reliability.evaluation import (
    Dataset,
    EvaluationResult,
    EvaluationRunner,
    EvaluationStatus,
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


class TestEvaluationResultBackwardCompatibility:
    """EvaluationResult.from_dict must load records written before ``status``,
    ``criterion``, and ``evaluator_config`` existed (Prompt 1/2 era)."""

    def test_old_style_successful_match_defaults_to_success(self):
        old_style = {
            "evaluator_name": "exact_match",
            "test_case_id": "t1",
            "score": 1.0,
            "passed": True,
            "label": "match",
        }
        result = EvaluationResult.from_dict(old_style)
        assert result.status == EvaluationStatus.SUCCESS

    def test_old_style_no_reference_result_is_inferred_as_skipped(self):
        old_style = {
            "evaluator_name": "exact_match",
            "test_case_id": "t1",
            "label": "no_reference",
        }
        result = EvaluationResult.from_dict(old_style)
        assert result.status == EvaluationStatus.SKIPPED

    def test_old_style_error_result_is_inferred_as_execution_error(self):
        old_style = {
            "evaluator_name": "raising_evaluator",
            "test_case_id": "t1",
            "error": "RuntimeError: boom",
        }
        result = EvaluationResult.from_dict(old_style)
        assert result.status == EvaluationStatus.EXECUTION_ERROR

    def test_old_style_result_has_empty_config_and_no_criterion(self):
        old_style = {"evaluator_name": "exact_match", "test_case_id": "t1", "label": "match"}
        result = EvaluationResult.from_dict(old_style)
        assert result.evaluator_config == {}
        assert result.criterion is None

    def test_new_style_explicit_status_is_respected_over_inference(self):
        new_style = {
            "evaluator_name": "exact_match",
            "test_case_id": "t1",
            "status": "invalid_configuration",
            "error": "some backend error",
        }
        result = EvaluationResult.from_dict(new_style)
        assert result.status == EvaluationStatus.INVALID_CONFIGURATION
