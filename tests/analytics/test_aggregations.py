"""
Tests for aggregation functions (evaluation, failure distribution,
retrieval, robustness, performance).
"""

from __future__ import annotations

from llm_reliability.analytics.aggregations import (
    aggregate_evaluation,
    aggregate_failures,
    aggregate_retrieval,
    aggregate_robustness,
    measure_performance,
)
from llm_reliability.diagnostics.diagnosis import ReliabilityDiagnostic
from llm_reliability.diagnostics.robustness import PerturbationType, RobustnessResult
from llm_reliability.diagnostics.taxonomy import DiagnosticStatus, FailureCategory
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
from llm_reliability.rag.evaluators import RetrievalEvaluationResult, RetrievalEvaluationStatus


def _create_mock_run() -> ExperimentRun:
    ds = Dataset(
        name="test", test_cases=[TestCase(id="tc-1", input="Q1", reference_answer="Paris")]
    )
    defn = ExperimentDefinition(
        experiment_id="exp-1",
        name="Test Exp",
        dataset_version=DatasetVersion(dataset=ds),
        model_config=ModelConfig(model_id="mock-model"),
        evaluator_configs=[
            EvaluatorConfig(evaluator_name="exact_match"),
            EvaluatorConfig(evaluator_name="faithfulness"),
        ],
    )
    tc_res_1 = TestCaseResult(
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
                score=1.0,
                passed=True,
                criterion="correctness",
                duration_seconds=0.01,
            ),
            EvaluationResult(
                evaluator_name="faithfulness",
                test_case_id="tc-1",
                status=EvaluationStatus.SUCCESS,
                score=1.0,
                passed=True,
                criterion="faithfulness",
                duration_seconds=0.02,
            ),
        ],
        duration_seconds=0.05,
    )
    tc_res_2 = TestCaseResult(
        test_case_id="tc-2",
        input_text="Q2",
        reference_answer="London",
        model_response=ModelResponse(
            test_case_id="tc-2", output_text="London", model_id="mock-model"
        ),
        evaluation_results=[
            EvaluationResult(
                evaluator_name="exact_match",
                test_case_id="tc-2",
                status=EvaluationStatus.SUCCESS,
                score=0.0,
                passed=False,
                criterion="correctness",
                duration_seconds=0.01,
            ),
            EvaluationResult(
                evaluator_name="faithfulness",
                test_case_id="tc-2",
                status=EvaluationStatus.SKIPPED,
                score=None,
                passed=None,
                criterion="faithfulness",
            ),
        ],
        duration_seconds=0.04,
    )
    run_res = RunResult(
        dataset_name="test",
        model_id="mock-model",
        evaluator_names=["exact_match", "faithfulness"],
        test_case_results=[tc_res_1, tc_res_2],
        duration_seconds=0.10,
    )
    return ExperimentRun(
        run_id="run-1",
        experiment_id="exp-1",
        status=RunStatus.SUCCEEDED,
        started_at="2026-11-01T10:00:00+00:00",
        finished_at="2026-11-01T10:00:02+00:00",
        dataset_version=defn.dataset_version,
        model_config=defn.model_config,
        evaluator_configs=defn.evaluator_configs,
        config_fingerprint="fp-123",
        result=run_res,
    )


def test_aggregate_evaluation() -> None:
    run = _create_mock_run()
    aggs = aggregate_evaluation(run)

    assert "exact_match" in aggs
    em = aggs["exact_match"]
    assert em.total_cases == 2
    assert em.num_evaluated == 2
    assert em.num_passed == 1
    assert em.num_failed == 1
    assert em.pass_rate_numerator == 1
    assert em.pass_rate_denominator == 2
    assert em.pass_rate == 0.5
    assert em.score_mean == 0.5

    assert "faithfulness" in aggs
    faith = aggs["faithfulness"]
    assert faith.num_evaluated == 2
    assert faith.num_success == 1
    assert faith.num_skipped == 1
    assert faith.num_passed == 1
    assert faith.num_failed == 0
    assert faith.pass_rate_numerator == 1
    assert faith.pass_rate_denominator == 1
    assert faith.pass_rate == 1.0


def test_aggregate_failures() -> None:
    diags = [
        ReliabilityDiagnostic(
            test_case_id="tc-1",
            question="Q1",
            generated_answer="A1",
            reference_answer="R1",
            retrieved_context=None,
            status=DiagnosticStatus.DETERMINED,
            failure_category=FailureCategory.CORRECT,
            explanation="Correct output.",
        ),
        ReliabilityDiagnostic(
            test_case_id="tc-2",
            question="Q2",
            generated_answer="A2",
            reference_answer="R2",
            retrieved_context=None,
            status=DiagnosticStatus.DETERMINED,
            failure_category=FailureCategory.CONTRADICTED,
            secondary_findings=[
                "evaluator_disagreement: ExactMatch vs LLMJudge",
                "insufficient_evidence: missing chunks",
            ],
            explanation="Contradicts retrieved context.",
        ),
        ReliabilityDiagnostic(
            test_case_id="tc-3",
            question="Q3",
            generated_answer="A3",
            reference_answer="R3",
            retrieved_context=None,
            status=DiagnosticStatus.DETERMINED,
            failure_category=FailureCategory.CONTRADICTED,
            secondary_findings=["evaluator_disagreement"],
            explanation="Contradicted again.",
        ),
    ]

    dist = aggregate_failures(diags, source_run_ids="run-1")
    assert dist.total_cases == 3
    assert dist.num_determined == 3
    assert dist.num_undetermined == 0
    assert dist.source_run_ids == ["run-1"]

    # Check primary
    assert "correct" in dist.primary_categories
    corr = dist.primary_categories["correct"]
    assert corr.count == 1
    assert corr.denominator == 3
    assert corr.proportion is not None
    assert round(corr.proportion, 3) == round(1 / 3, 3)

    assert "contradicted" in dist.primary_categories
    contra = dist.primary_categories["contradicted"]
    assert contra.count == 2
    assert contra.denominator == 3
    assert contra.proportion is not None
    assert round(contra.proportion, 3) == round(2 / 3, 3)

    # Check secondary findings preserved independently
    assert "evaluator_disagreement" in dist.secondary_findings
    disagree = dist.secondary_findings["evaluator_disagreement"]
    assert disagree.count == 2
    assert disagree.denominator == 3

    assert "insufficient_evidence" in dist.secondary_findings
    insuff = dist.secondary_findings["insufficient_evidence"]
    assert insuff.count == 1
    assert insuff.denominator == 3


def test_aggregate_retrieval() -> None:
    ret_results = [
        RetrievalEvaluationResult(
            test_case_id="tc-1",
            evaluator_name="hit_at_1",
            metric_name="hit_at_k",
            evaluator_config={"k": 1},
            score=1.0,
            status=RetrievalEvaluationStatus.SUCCESS,
        ),
        RetrievalEvaluationResult(
            test_case_id="tc-2",
            evaluator_name="hit_at_1",
            metric_name="hit_at_k",
            evaluator_config={"k": 1},
            score=0.0,
            status=RetrievalEvaluationStatus.SUCCESS,
        ),
        RetrievalEvaluationResult(
            test_case_id="tc-1",
            evaluator_name="hit_at_3",
            metric_name="hit_at_k",
            evaluator_config={"k": 3},
            score=1.0,
            status=RetrievalEvaluationStatus.SUCCESS,
        ),
        RetrievalEvaluationResult(
            test_case_id="tc-2",
            evaluator_name="hit_at_3",
            metric_name="hit_at_k",
            evaluator_config={"k": 3},
            score=1.0,
            status=RetrievalEvaluationStatus.SUCCESS,
        ),
        RetrievalEvaluationResult(
            test_case_id="tc-3",
            evaluator_name="hit_at_3",
            metric_name="hit_at_k",
            evaluator_config={"k": 3},
            score=None,
            status=RetrievalEvaluationStatus.SKIPPED,
        ),
    ]

    sums = aggregate_retrieval(ret_results)
    assert len(sums) == 2

    hit1 = next(s for s in sums if s.metric_name == "hit_at_k" and s.k == 1)
    assert hit1.evaluated_cases == 2
    assert hit1.mean_score == 0.5

    hit3 = next(s for s in sums if s.metric_name == "hit_at_k" and s.k == 3)
    assert hit3.evaluated_cases == 2
    assert hit3.skipped_cases == 1
    assert hit3.mean_score == 1.0


def test_aggregate_robustness() -> None:
    results = [
        RobustnessResult(
            baseline_test_case_id="tc-1",
            perturbed_test_case_id="tc-1-typo",
            perturbation_type=PerturbationType.QUESTION_REWORDING,
            baseline_category="correct",
            perturbed_category="correct",
            outcome_changed=False,
            failure_category_transition=None,
        ),
        RobustnessResult(
            baseline_test_case_id="tc-2",
            perturbed_test_case_id="tc-2-typo",
            perturbation_type=PerturbationType.QUESTION_REWORDING,
            baseline_category="correct",
            perturbed_category="incorrect",
            outcome_changed=True,
            failure_category_transition="CORRECT->INCORRECT",
        ),
    ]

    summary = aggregate_robustness(results)
    assert summary.total_pairs == 2
    assert summary.num_unchanged == 1
    assert summary.num_changed == 1
    assert summary.consistency_rate == 0.5
    assert summary.transition_counts.get("CORRECT->INCORRECT") == 1
    assert "question_rewording" in summary.by_perturbation
    assert summary.by_perturbation["question_rewording"]["rate"] == 0.5


def test_measure_performance() -> None:
    run = _create_mock_run()
    perf = measure_performance(run)

    assert perf.run_id == "run-1"
    assert perf.wall_clock_duration_seconds == 2.0
    assert perf.per_case_duration_seconds_mean == 0.045
    assert perf.tokens_available is False
    assert perf.cost_available is False
    assert any("cost" in n.lower() for n in perf.notes)
