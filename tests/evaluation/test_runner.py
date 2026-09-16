"""Tests for EvaluationRunner orchestration and failure isolation."""

import pytest

from llm_reliability.evaluation import (
    Dataset,
    EvaluationResult,
    EvaluationRunner,
    Evaluator,
    ExactMatchEvaluator,
    MockAdapter,
    ModelConfig,
    ModelResponse,
    TestCase,
)


class RaisingEvaluator(Evaluator):
    """Test-only evaluator that always raises, used to verify error isolation."""

    name = "raising_evaluator"

    def evaluate(self, test_case: TestCase, response: ModelResponse) -> EvaluationResult:
        raise RuntimeError("evaluator exploded")


def _dataset(*test_cases: TestCase) -> Dataset:
    return Dataset(name="test-dataset", test_cases=list(test_cases))


def _config() -> ModelConfig:
    return ModelConfig(model_id="mock-adapter-v1")


class TestSingleTestCaseExecution:
    def test_successful_single_test_case(self):
        dataset = _dataset(TestCase(id="t1", input="2+2?", reference_answer="4"))
        runner = EvaluationRunner(
            adapter=MockAdapter(responses={"t1": "4"}), evaluators=[ExactMatchEvaluator()]
        )
        run_result = runner.run(dataset, _config())

        assert run_result.num_test_cases == 1
        assert run_result.num_succeeded == 1
        assert run_result.num_failed == 0

        result = run_result.test_case_results[0]
        assert result.test_case_id == "t1"
        assert result.succeeded is True
        assert result.model_response.output_text == "4"
        assert result.evaluation_results[0].passed is True


class TestMultiTestCaseExecution:
    def test_preserves_order_and_ids(self):
        dataset = _dataset(
            TestCase(id="t1", input="a"),
            TestCase(id="t2", input="b"),
            TestCase(id="t3", input="c"),
        )
        runner = EvaluationRunner(adapter=MockAdapter(), evaluators=[ExactMatchEvaluator()])
        run_result = runner.run(dataset, _config())

        assert [r.test_case_id for r in run_result.test_case_results] == ["t1", "t2", "t3"]

    def test_multiple_evaluators_all_run(self):
        dataset = _dataset(TestCase(id="t1", input="hello", reference_answer="hello"))
        runner = EvaluationRunner(
            adapter=MockAdapter(),
            evaluators=[ExactMatchEvaluator(), ExactMatchEvaluator()],
        )
        run_result = runner.run(dataset, _config())

        assert run_result.evaluator_names == ["exact_match", "exact_match"]
        assert len(run_result.test_case_results[0].evaluation_results) == 2


class TestFailureIsolation:
    def test_adapter_error_isolated_to_one_test_case(self):
        dataset = _dataset(
            TestCase(id="t1", input="a", reference_answer="a"),
            TestCase(id="t2", input="b", reference_answer="b"),
        )
        adapter = MockAdapter(errors={"t1": "simulated failure"})
        runner = EvaluationRunner(adapter=adapter, evaluators=[ExactMatchEvaluator()])
        run_result = runner.run(dataset, _config())

        assert run_result.num_succeeded == 1
        assert run_result.num_failed == 1

        failed, ok = run_result.test_case_results
        assert failed.succeeded is False
        assert "simulated failure" in failed.execution_error
        assert failed.model_response is None
        assert failed.evaluation_results == []

        assert ok.succeeded is True
        assert ok.evaluation_results[0].passed is True

    def test_evaluator_error_isolated_and_recorded(self):
        dataset = _dataset(TestCase(id="t1", input="a", reference_answer="a"))
        runner = EvaluationRunner(adapter=MockAdapter(), evaluators=[RaisingEvaluator()])
        run_result = runner.run(dataset, _config())

        result = run_result.test_case_results[0]
        assert result.succeeded is True
        assert result.model_response is not None
        eval_result = result.evaluation_results[0]
        assert eval_result.error is not None
        assert "evaluator exploded" in eval_result.error
        assert eval_result.evaluator_name == "raising_evaluator"

    def test_one_evaluator_failing_does_not_block_others(self):
        dataset = _dataset(TestCase(id="t1", input="a", reference_answer="a"))
        runner = EvaluationRunner(
            adapter=MockAdapter(), evaluators=[RaisingEvaluator(), ExactMatchEvaluator()]
        )
        run_result = runner.run(dataset, _config())

        results = run_result.test_case_results[0].evaluation_results
        assert results[0].error is not None
        assert results[1].error is None
        assert results[1].passed is True

    def test_failure_in_one_test_case_does_not_affect_others_in_batch(self):
        dataset = _dataset(
            TestCase(id="t1", input="a", reference_answer="a"),
            TestCase(id="t2", input="b", reference_answer="b"),
            TestCase(id="t3", input="c", reference_answer="c"),
        )
        adapter = MockAdapter(errors={"t2": "boom"})
        runner = EvaluationRunner(adapter=adapter, evaluators=[ExactMatchEvaluator()])
        run_result = runner.run(dataset, _config())

        statuses = {r.test_case_id: r.succeeded for r in run_result.test_case_results}
        assert statuses == {"t1": True, "t2": False, "t3": True}


class TestRunnerConstruction:
    def test_requires_at_least_one_evaluator(self):
        with pytest.raises(ValueError, match="at least one evaluator"):
            EvaluationRunner(adapter=MockAdapter(), evaluators=[])


class TestRunResultDiagnostics:
    def test_run_result_exposes_dataset_and_model_identity(self):
        dataset = _dataset(TestCase(id="t1", input="a"))
        runner = EvaluationRunner(adapter=MockAdapter(), evaluators=[ExactMatchEvaluator()])
        run_result = runner.run(dataset, ModelConfig(model_id="mock-adapter-v1"))

        assert run_result.dataset_name == "test-dataset"
        assert run_result.model_id == "mock-adapter-v1"
