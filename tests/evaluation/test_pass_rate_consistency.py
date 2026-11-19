"""
Regression tests for evaluation integrity, pass-rate consistency, and separation
of execution success from evaluation correctness.
"""

from __future__ import annotations

import pytest

from llm_reliability.api.schemas.runs import RunDetailResponse, TestCaseResultResponse
from llm_reliability.api.services import to_run_detail, to_run_summary
from llm_reliability.evaluation.adapters import MockAdapter
from llm_reliability.evaluation.evaluators import ExactMatchEvaluator
from llm_reliability.evaluation.models import (
    Dataset,
    EvaluationResult,
    EvaluationStatus,
    ModelConfig,
    ModelResponse,
    RunResult,
    TestCase,
    TestCaseResult,
)
from llm_reliability.evaluation.runner import EvaluationRunner
from llm_reliability.experiments.models import DatasetVersion, ExperimentDefinition, ExperimentRun, RunStatus


def test_exact_match_correct_answer() -> None:
    """Exact match passing test case has succeeded=True and passed=True."""
    dataset = Dataset(
        name="test-ds",
        test_cases=[TestCase(id="t1", input="What is the capital of France?", reference_answer="Paris")],
    )
    adapter = MockAdapter(responses={"t1": "Paris"})
    runner = EvaluationRunner(adapter=adapter, evaluators=[ExactMatchEvaluator()])
    run_result = runner.run(dataset, ModelConfig(model_id="test-model"))

    assert len(run_result.test_case_results) == 1
    tc = run_result.test_case_results[0]
    assert tc.succeeded is True
    assert tc.passed is True
    assert run_result.num_succeeded == 1
    assert run_result.num_passed == 1
    assert run_result.pass_rate == 1.0


def test_exact_match_with_extra_text_fails_evaluation_while_execution_succeeds() -> None:
    """Extra prompt/context text causes exact match to fail (passed=False)

    even though model inference succeeded (succeeded=True).
    """
    dataset = Dataset(
        name="qa-geography-baseline-v1",
        test_cases=[TestCase(id="tc-1", input="What is the capital of France?", reference_answer="Paris")],
    )
    # Model generates the answer but echoes the prompt and context text
    noisy_output = "Question: What is the capital of France?\nAnswer: Paris"
    adapter = MockAdapter(responses={"tc-1": noisy_output})
    runner = EvaluationRunner(adapter=adapter, evaluators=[ExactMatchEvaluator()])
    run_result = runner.run(dataset, ModelConfig(model_id="mock-adapter-v1"))

    assert len(run_result.test_case_results) == 1
    tc = run_result.test_case_results[0]

    # Execution succeeded (response was produced without runtime exception)
    assert tc.succeeded is True
    assert tc.execution_error is None
    assert tc.model_response is not None
    assert tc.model_response.output_text == noisy_output

    # Evaluator verdict is fail
    assert len(tc.evaluation_results) == 1
    eval_res = tc.evaluation_results[0]
    assert eval_res.evaluator_name == "exact_match"
    assert eval_res.status == EvaluationStatus.SUCCESS
    assert eval_res.passed is False
    assert eval_res.score == 0.0
    assert eval_res.label == "mismatch"

    # Crucial assertion: test case passed property MUST be False!
    assert tc.passed is False

    # Run-level aggregation must strictly distinguish execution from pass rate
    assert run_result.num_succeeded == 1
    assert run_result.num_failed == 0
    assert run_result.num_passed == 0
    assert run_result.num_eval_failed == 1
    assert run_result.pass_rate == 0.0


def test_incorrect_answer_fails_evaluation() -> None:
    """Completely incorrect answer fails evaluation (succeeded=True, passed=False)."""
    dataset = Dataset(
        name="test-ds",
        test_cases=[TestCase(id="t1", input="What is the capital of France?", reference_answer="Paris")],
    )
    adapter = MockAdapter(responses={"t1": "London"})
    runner = EvaluationRunner(adapter=adapter, evaluators=[ExactMatchEvaluator()])
    run_result = runner.run(dataset, ModelConfig(model_id="test-model"))

    tc = run_result.test_case_results[0]
    assert tc.succeeded is True
    assert tc.passed is False
    assert run_result.num_succeeded == 1
    assert run_result.num_passed == 0
    assert run_result.pass_rate == 0.0


def test_failed_execution_fails_both_succeeded_and_passed() -> None:
    """Generation exception results in succeeded=False and passed=False."""
    dataset = Dataset(
        name="test-ds",
        test_cases=[TestCase(id="t1", input="question", reference_answer="answer")],
    )
    adapter = MockAdapter(errors={"t1": "API timeout connecting to upstream provider"})
    runner = EvaluationRunner(adapter=adapter, evaluators=[ExactMatchEvaluator()])
    run_result = runner.run(dataset, ModelConfig(model_id="test-model"))

    tc = run_result.test_case_results[0]
    assert tc.succeeded is False
    assert tc.passed is False
    assert tc.execution_error is not None
    assert run_result.num_succeeded == 0
    assert run_result.num_failed == 1
    assert run_result.num_passed == 0
    assert run_result.num_eval_failed == 1
    assert run_result.pass_rate == 0.0


def test_multi_evaluator_any_fail_rule() -> None:
    """When multiple evaluators disagree (one passes, one fails), test case does not pass."""
    tc = TestCaseResult(
        test_case_id="t1",
        input_text="question",
        reference_answer="Paris",
        model_response=ModelResponse(test_case_id="t1", output_text="Paris", model_id="m"),
        evaluation_results=[
            EvaluationResult(
                evaluator_name="eval_a",
                test_case_id="t1",
                status=EvaluationStatus.SUCCESS,
                passed=True,
                score=1.0,
            ),
            EvaluationResult(
                evaluator_name="eval_b",
                test_case_id="t1",
                status=EvaluationStatus.SUCCESS,
                passed=False,
                score=0.0,
            ),
        ],
        execution_error=None,
    )
    assert tc.succeeded is True
    assert tc.passed is False


def test_multi_evaluator_all_pass() -> None:
    """When all evaluators pass, test case passed is True."""
    tc = TestCaseResult(
        test_case_id="t1",
        input_text="question",
        reference_answer="Paris",
        model_response=ModelResponse(test_case_id="t1", output_text="Paris", model_id="m"),
        evaluation_results=[
            EvaluationResult(
                evaluator_name="eval_a",
                test_case_id="t1",
                status=EvaluationStatus.SUCCESS,
                passed=True,
                score=1.0,
            ),
            EvaluationResult(
                evaluator_name="eval_b",
                test_case_id="t1",
                status=EvaluationStatus.SUCCESS,
                passed=True,
                score=0.98,
            ),
        ],
        execution_error=None,
    )
    assert tc.succeeded is True
    assert tc.passed is True


def test_evaluator_error_status_does_not_pass() -> None:
    """An evaluator with an execution or validation error status prevents passing."""
    tc = TestCaseResult(
        test_case_id="t1",
        input_text="question",
        reference_answer="Paris",
        model_response=ModelResponse(test_case_id="t1", output_text="Paris", model_id="m"),
        evaluation_results=[
            EvaluationResult(
                evaluator_name="eval_err",
                test_case_id="t1",
                status=EvaluationStatus.EXECUTION_ERROR,
                passed=None,
                error="Judge crashed",
            )
        ],
        execution_error=None,
    )
    assert tc.succeeded is True
    assert tc.passed is False


def test_run_aggregation_mixed_results() -> None:
    """RunResult aggregates execution and evaluation numbers accurately."""
    results = [
        TestCaseResult(
            test_case_id="t1",
            input_text="q1",
            reference_answer="a1",
            model_response=ModelResponse(test_case_id="t1", output_text="a1", model_id="m"),
            evaluation_results=[
                EvaluationResult(
                    evaluator_name="e", test_case_id="t1", status=EvaluationStatus.SUCCESS, passed=True
                )
            ],
            execution_error=None,
        ),
        TestCaseResult(
            test_case_id="t2",
            input_text="q2",
            reference_answer="a2",
            model_response=ModelResponse(test_case_id="t2", output_text="wrong", model_id="m"),
            evaluation_results=[
                EvaluationResult(
                    evaluator_name="e", test_case_id="t2", status=EvaluationStatus.SUCCESS, passed=False
                )
            ],
            execution_error=None,
        ),
        TestCaseResult(
            test_case_id="t3",
            input_text="q3",
            reference_answer="a3",
            model_response=None,
            evaluation_results=[],
            execution_error="Network timeout",
        ),
    ]
    run_result = RunResult(
        dataset_name="mixed-ds",
        model_id="m",
        evaluator_names=["e"],
        test_case_results=results,
    )
    assert run_result.num_test_cases == 3
    assert run_result.num_succeeded == 2
    assert run_result.num_failed == 1
    assert run_result.num_passed == 1
    assert run_result.num_eval_failed == 2
    assert pytest.approx(run_result.pass_rate, 0.01) == 1 / 3


def test_api_serialization_preserves_passed_distinction() -> None:
    """to_run_detail and to_run_summary correctly serialize passed vs succeeded."""
    dataset = Dataset(
        name="qa-geography-baseline-v1",
        test_cases=[
            TestCase(id="tc-1", input="What is the capital of France?", reference_answer="Paris"),
            TestCase(id="tc-2", input="Capital of Germany?", reference_answer="Berlin"),
        ],
    )
    # tc-1 fails evaluation due to extra text; tc-2 passes
    results = [
        TestCaseResult(
            test_case_id="tc-1",
            input_text="What is the capital of France?",
            reference_answer="Paris",
            model_response=ModelResponse(
                test_case_id="tc-1",
                output_text="Question: What is the capital of France?\nAnswer: Paris",
                model_id="mock-adapter-v1",
            ),
            evaluation_results=[
                EvaluationResult(
                    evaluator_name="exact_match",
                    test_case_id="tc-1",
                    status=EvaluationStatus.SUCCESS,
                    passed=False,
                    score=0.0,
                )
            ],
            execution_error=None,
        ),
        TestCaseResult(
            test_case_id="tc-2",
            input_text="Capital of Germany?",
            reference_answer="Berlin",
            model_response=ModelResponse(
                test_case_id="tc-2",
                output_text="Berlin",
                model_id="mock-adapter-v1",
            ),
            evaluation_results=[
                EvaluationResult(
                    evaluator_name="exact_match",
                    test_case_id="tc-2",
                    status=EvaluationStatus.SUCCESS,
                    passed=True,
                    score=1.0,
                )
            ],
            execution_error=None,
        ),
    ]
    run_result = RunResult(
        dataset_name="qa-geography-baseline-v1",
        model_id="mock-adapter-v1",
        evaluator_names=["exact_match"],
        test_case_results=results,
    )
    run = ExperimentRun(
        run_id="run-test-eval-integrity",
        experiment_id="exp-test",
        status=RunStatus.SUCCEEDED,
        started_at="2026-09-16T15:00:00Z",
        finished_at="2026-09-16T15:00:02Z",
        dataset_version=DatasetVersion(dataset=dataset),
        model_config=ModelConfig(model_id="mock-adapter-v1"),
        evaluator_configs=[],
        config_fingerprint="fp123",
        result=run_result,
    )

    detail: RunDetailResponse = to_run_detail(run)
    assert detail.num_test_cases == 2
    assert detail.num_succeeded == 2
    assert detail.num_failed == 0
    assert detail.num_passed == 1
    assert detail.pass_rate == 0.5

    # Check individual cases in detail
    by_id = {tc.test_case_id: tc for tc in detail.test_case_results}
    assert by_id["tc-1"].succeeded is True
    assert by_id["tc-1"].passed is False  # MUST be False!

    assert by_id["tc-2"].succeeded is True
    assert by_id["tc-2"].passed is True

    summary = to_run_summary(run)
    assert summary.num_test_cases == 2
    assert summary.num_succeeded == 2
    assert summary.num_passed == 1
    assert summary.pass_rate == 0.5
