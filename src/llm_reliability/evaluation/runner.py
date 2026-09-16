"""
Evaluation run orchestration.

``EvaluationRunner`` executes a dataset against a single model/application
adapter, applies one or more evaluators to each resulting response, and
returns a structured ``RunResult``. Failures in generation or evaluation
are isolated to the affected test case and recorded explicitly on it;
they never abort the run or corrupt other test cases' results.
"""

from __future__ import annotations

from collections.abc import Sequence

from llm_reliability.evaluation.adapters import ModelAdapter
from llm_reliability.evaluation.evaluators import Evaluator
from llm_reliability.evaluation.models import (
    Dataset,
    EvaluationResult,
    EvaluationStatus,
    ExecutionRequest,
    ModelConfig,
    ModelResponse,
    RunResult,
    TestCase,
    TestCaseResult,
)


class EvaluationRunner:
    """Executes a dataset against an adapter and a fixed set of evaluators."""

    def __init__(self, adapter: ModelAdapter, evaluators: Sequence[Evaluator]) -> None:
        if len(evaluators) == 0:
            raise ValueError("EvaluationRunner requires at least one evaluator.")
        self._adapter = adapter
        self._evaluators = list(evaluators)

    def run(self, dataset: Dataset, model_config: ModelConfig) -> RunResult:
        """Execute every test case in ``dataset`` and return a structured run result.

        Test case order is preserved in ``RunResult.test_case_results``.
        """
        test_case_results = [
            self._run_single_test_case(test_case, model_config) for test_case in dataset.test_cases
        ]
        return RunResult(
            dataset_name=dataset.name,
            model_id=model_config.model_id,
            evaluator_names=[evaluator.name for evaluator in self._evaluators],
            test_case_results=test_case_results,
        )

    def _run_single_test_case(
        self, test_case: TestCase, model_config: ModelConfig
    ) -> TestCaseResult:
        request = ExecutionRequest(
            test_case_id=test_case.id,
            input_text=test_case.input,
            model_config=model_config,
        )
        try:
            response = self._adapter.generate(request)
        except Exception as exc:
            return TestCaseResult(
                test_case_id=test_case.id,
                input_text=test_case.input,
                reference_answer=test_case.reference_answer,
                model_response=None,
                evaluation_results=[],
                execution_error=f"{type(exc).__name__}: {exc}",
            )

        evaluation_results = [
            self._run_single_evaluator(evaluator, test_case, response)
            for evaluator in self._evaluators
        ]
        return TestCaseResult(
            test_case_id=test_case.id,
            input_text=test_case.input,
            reference_answer=test_case.reference_answer,
            model_response=response,
            evaluation_results=evaluation_results,
            execution_error=None,
        )

    @staticmethod
    def _run_single_evaluator(
        evaluator: Evaluator, test_case: TestCase, response: ModelResponse
    ) -> EvaluationResult:
        try:
            return evaluator.evaluate(test_case, response)
        except Exception as exc:
            return EvaluationResult(
                evaluator_name=getattr(evaluator, "name", evaluator.__class__.__name__),
                test_case_id=test_case.id,
                status=EvaluationStatus.EXECUTION_ERROR,
                criterion=getattr(evaluator, "criterion", None),
                error=f"{type(exc).__name__}: {exc}",
            )
