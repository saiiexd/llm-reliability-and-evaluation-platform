"""
Tests for evaluator agreement and disagreement analytics.
"""

from __future__ import annotations

from llm_reliability.analytics.agreement import (
    compute_all_pairwise_agreements,
    compute_evaluator_agreement,
)
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
from llm_reliability.experiments.models import (
    DatasetVersion,
    EvaluatorConfig,
    ExperimentDefinition,
    ExperimentRun,
    RunStatus,
)


def _build_agreement_run() -> ExperimentRun:
    ds = Dataset(
        name="test", test_cases=[TestCase(id="tc-1", input="Q1", reference_answer="Paris")]
    )
    defn = ExperimentDefinition(
        experiment_id="exp-agree",
        name="Agreement Test",
        dataset_version=DatasetVersion(dataset=ds),
        model_config=ModelConfig(model_id="mock-model"),
        evaluator_configs=[
            EvaluatorConfig(evaluator_name="exact_match"),
            EvaluatorConfig(evaluator_name="llm_judge"),
            EvaluatorConfig(evaluator_name="faithfulness"),
        ],
    )

    tc_results = [
        TestCaseResult(
            test_case_id="tc-1",
            input_text="Q1",
            reference_answer="Paris",
            model_response=ModelResponse(
                test_case_id="tc-1", output_text="Paris", model_id="mock-model"
            ),
            evaluation_results=[
                EvaluationResult(
                    evaluator_name="exact_match",
                    test_case_id="tc-1",
                    status=EvaluationStatus.SUCCESS,
                    passed=True,
                    criterion="correctness",
                ),
                EvaluationResult(
                    evaluator_name="llm_judge",
                    test_case_id="tc-1",
                    status=EvaluationStatus.SUCCESS,
                    passed=True,
                    criterion="correctness",
                ),
                EvaluationResult(
                    evaluator_name="faithfulness",
                    test_case_id="tc-1",
                    status=EvaluationStatus.SUCCESS,
                    passed=True,
                    criterion="context_grounding",
                ),
            ],
        ),
        TestCaseResult(
            test_case_id="tc-2",
            input_text="Q2",
            reference_answer="Paris",
            model_response=ModelResponse(
                test_case_id="tc-2", output_text="Capital is Paris", model_id="mock-model"
            ),
            evaluation_results=[
                EvaluationResult(
                    evaluator_name="exact_match",
                    test_case_id="tc-2",
                    status=EvaluationStatus.SUCCESS,
                    passed=False,
                    criterion="correctness",
                ),
                EvaluationResult(
                    evaluator_name="llm_judge",
                    test_case_id="tc-2",
                    status=EvaluationStatus.SUCCESS,
                    passed=True,
                    criterion="correctness",
                ),
                EvaluationResult(
                    evaluator_name="faithfulness",
                    test_case_id="tc-2",
                    status=EvaluationStatus.SUCCESS,
                    passed=True,
                    criterion="context_grounding",
                ),
            ],
        ),
    ]

    return ExperimentRun(
        run_id="run-agree",
        experiment_id="exp-agree",
        status=RunStatus.SUCCEEDED,
        started_at="2026-11-01T10:00:00Z",
        finished_at="2026-11-01T10:00:01Z",
        dataset_version=defn.dataset_version,
        model_config=defn.model_config,
        evaluator_configs=defn.evaluator_configs,
        config_fingerprint="fp-agree",
        result=RunResult(
            dataset_name="test",
            model_id="mock-model",
            evaluator_names=["exact_match", "llm_judge", "faithfulness"],
            test_case_results=tc_results,
        ),
    )


def test_evaluator_agreement_compatible_criteria() -> None:
    run = _build_agreement_run()
    summary = compute_evaluator_agreement(run, "exact_match", "llm_judge")

    assert summary is not None
    assert summary.criterion == "correctness"
    assert summary.cases_compared == 2
    assert summary.agreements == 1
    assert summary.disagreements == 1
    assert summary.agreement_rate == 0.5
    assert summary.disagreement_test_case_ids == ["tc-2"]


def test_evaluator_agreement_incompatible_criteria_rejected() -> None:
    run = _build_agreement_run()
    # exact_match evaluates "correctness", faithfulness evaluates "context_grounding"
    summary = compute_evaluator_agreement(run, "exact_match", "faithfulness")
    assert summary is None


def test_compute_all_pairwise_agreements() -> None:
    run = _build_agreement_run()
    summaries = compute_all_pairwise_agreements(run)

    # Only exact_match vs llm_judge have compatible criteria ("correctness")
    assert len(summaries) == 1
    assert summaries[0].evaluator_a == "exact_match"
    assert summaries[0].evaluator_b == "llm_judge"
    assert summaries[0].disagreements == 1
