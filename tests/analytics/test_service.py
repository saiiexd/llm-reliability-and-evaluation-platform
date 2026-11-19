"""
Tests for AnalyticsService query and overview operations.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from llm_reliability.analytics.service import AnalyticsService
from llm_reliability.diagnostics.diagnosis import ReliabilityDiagnostic
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
from llm_reliability.experiments import LocalFileExperimentRepository
from llm_reliability.experiments.models import (
    DatasetVersion,
    EvaluatorConfig,
    ExperimentDefinition,
    ExperimentRun,
    RunStatus,
)
from llm_reliability.rag.evaluators import RetrievalEvaluationResult, RetrievalEvaluationStatus


def _setup_service() -> tuple[AnalyticsService, str, str]:
    tmp_dir = Path(tempfile.mkdtemp())
    repo = LocalFileExperimentRepository(tmp_dir)

    ds = Dataset(
        name="service-ds", test_cases=[TestCase(id="tc-1", input="Q1", reference_answer="Paris")]
    )
    defn = ExperimentDefinition(
        experiment_id="exp-service",
        name="Service Test Exp",
        dataset_version=DatasetVersion(dataset=ds),
        model_config=ModelConfig(model_id="mock-model"),
        evaluator_configs=[EvaluatorConfig(evaluator_name="exact_match")],
    )
    repo.save_experiment(defn)

    tc = TestCaseResult(
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
            )
        ],
        duration_seconds=0.05,
    )
    run1 = ExperimentRun(
        run_id="run-service-1",
        experiment_id="exp-service",
        status=RunStatus.SUCCEEDED,
        started_at="2026-11-01T10:00:00Z",
        finished_at="2026-11-01T10:00:01Z",
        dataset_version=defn.dataset_version,
        model_config=defn.model_config,
        evaluator_configs=defn.evaluator_configs,
        config_fingerprint=defn.config_fingerprint,
        result=RunResult(
            dataset_name="service-ds",
            model_id="mock-model",
            evaluator_names=["exact_match"],
            test_case_results=[tc],
        ),
    )
    repo.save_run(run1)

    tc2 = TestCaseResult(
        test_case_id="tc-1",
        input_text="Q1",
        reference_answer="Paris",
        model_response=ModelResponse(
            test_case_id="tc-1", output_text="London", model_id="mock-model"
        ),
        evaluation_results=[
            EvaluationResult(
                evaluator_name="exact_match",
                test_case_id="tc-1",
                status=EvaluationStatus.SUCCESS,
                score=0.0,
                passed=False,
                criterion="correctness",
            )
        ],
        duration_seconds=0.06,
    )
    run2 = ExperimentRun(
        run_id="run-service-2",
        experiment_id="exp-service",
        status=RunStatus.SUCCEEDED,
        started_at="2026-11-01T11:00:00Z",
        finished_at="2026-11-01T11:00:01Z",
        dataset_version=defn.dataset_version,
        model_config=defn.model_config,
        evaluator_configs=defn.evaluator_configs,
        config_fingerprint=defn.config_fingerprint,
        result=RunResult(
            dataset_name="service-ds",
            model_id="mock-model",
            evaluator_names=["exact_match"],
            test_case_results=[tc2],
        ),
    )
    repo.save_run(run2)

    diag_map = {
        "run-service-1": [
            ReliabilityDiagnostic(
                test_case_id="tc-1",
                question="Q1",
                generated_answer="Paris",
                reference_answer="Paris",
                retrieved_context=None,
                status=DiagnosticStatus.DETERMINED,
                failure_category=FailureCategory.CORRECT,
                explanation="Match.",
            )
        ],
        "run-service-2": [
            ReliabilityDiagnostic(
                test_case_id="tc-1",
                question="Q1",
                generated_answer="London",
                reference_answer="Paris",
                retrieved_context=None,
                status=DiagnosticStatus.DETERMINED,
                failure_category=FailureCategory.INCORRECT,
                explanation="Mismatch.",
            )
        ],
    }

    ret_map = {
        "run-service-1": [
            RetrievalEvaluationResult(
                test_case_id="tc-1",
                evaluator_name="mrr_eval",
                metric_name="mrr",
                evaluator_config={},
                score=1.0,
                status=RetrievalEvaluationStatus.SUCCESS,
            )
        ]
    }

    service = AnalyticsService(
        repository=repo,
        diagnostics_loader=lambda r_id: diag_map.get(r_id, []),
        retrieval_loader=lambda r_id: ret_map.get(r_id, []),
    )
    return service, "run-service-1", "run-service-2"


def test_service_overview() -> None:
    service, r1, r2 = _setup_service()
    overview = service.get_overview()

    assert overview.total_experiments == 1
    assert overview.total_runs == 2
    assert overview.total_test_cases == 2
    assert overview.total_evaluations == 2
    assert overview.evaluator_coverage.get("exact_match") == 2
    assert overview.primary_failure_distribution.get("correct") == 1
    assert overview.primary_failure_distribution.get("incorrect") == 1
    assert overview.retrieval_runs_count == 1
    assert overview.mean_retrieval_mrr == 1.0


def test_service_queries() -> None:
    service, r1, r2 = _setup_service()

    # Evaluation aggregates
    agg = service.get_run_evaluation_aggregates(r1)
    assert "exact_match" in agg
    assert agg["exact_match"].pass_rate == 1.0

    # Failure distribution
    f_dist = service.get_run_failures(r1)
    assert f_dist.total_cases == 1
    assert f_dist.primary_categories["correct"].category == FailureCategory.CORRECT

    # Performance
    perf = service.get_run_performance(r1)
    assert perf.tokens_available is False

    # Comparison
    comp = service.compare([r1, r2])
    assert comp.compatibility.value == "identical_configuration"
    assert comp.evaluation_comparisons["exact_match"]["pass_rates"][r1] == 1.0
    assert comp.evaluation_comparisons["exact_match"]["pass_rates"][r2] == 0.0

    # Trend
    trend = service.get_trend(experiment_id="exp-service")
    assert trend.has_trend is True
    assert trend.run_ids == [r1, r2]
