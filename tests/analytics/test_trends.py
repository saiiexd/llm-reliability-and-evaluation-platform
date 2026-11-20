"""
Tests for trend analysis across sequential runs.
"""

from __future__ import annotations

from llm_reliability.analytics.trends import compute_trend_analysis
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


def _make_trend_run(run_id: str, timestamp: str, fingerprint: str, score: float) -> ExperimentRun:
    ds = Dataset(name="trend", test_cases=[TestCase(id="tc-1", input="Q1", reference_answer="A1")])
    defn = ExperimentDefinition(
        experiment_id="exp-trend",
        name="Trend Exp",
        dataset_version=DatasetVersion(dataset=ds),
        model_config=ModelConfig(model_id="mock-model"),
        evaluator_configs=[EvaluatorConfig(evaluator_name="exact_match")],
    )
    tc = TestCaseResult(
        test_case_id="tc-1",
        input_text="Q1",
        reference_answer="A1",
        model_response=ModelResponse(
            test_case_id="tc-1", output_text="output", model_id="mock-model"
        ),
        evaluation_results=[
            EvaluationResult(
                evaluator_name="exact_match",
                test_case_id="tc-1",
                status=EvaluationStatus.SUCCESS,
                score=score,
                passed=(score > 0.5),
                criterion="correctness",
            )
        ],
        duration_seconds=0.1,
    )
    return ExperimentRun(
        run_id=run_id,
        experiment_id="exp-trend",
        status=RunStatus.SUCCEEDED,
        started_at=timestamp,
        finished_at=timestamp,
        dataset_version=defn.dataset_version,
        model_config=defn.model_config,
        evaluator_configs=defn.evaluator_configs,
        config_fingerprint=fingerprint,
        result=RunResult(
            dataset_name="trend",
            model_id="mock-model",
            evaluator_names=["exact_match"],
            test_case_results=[tc],
        ),
    )


def test_trend_empty_and_single_run() -> None:
    empty_trend = compute_trend_analysis([])
    assert empty_trend.has_trend is False
    assert len(empty_trend.run_ids) == 0

    single_run = _make_trend_run("run-1", "2026-11-01T10:00:00Z", "fp-1", 1.0)
    single_trend = compute_trend_analysis([single_run])
    assert single_trend.has_trend is False
    assert len(single_trend.run_ids) == 1


def test_trend_multi_run_chronological_and_transitions() -> None:
    # Deliberately feed out of chronological order
    r2 = _make_trend_run("run-2", "2026-11-01T11:00:00Z", "fp-1", 0.5)
    r1 = _make_trend_run("run-1", "2026-11-01T10:00:00Z", "fp-1", 1.0)
    r3 = _make_trend_run("run-3", "2026-11-01T12:00:00Z", "fp-2", 0.8)

    trend = compute_trend_analysis([r2, r1, r3], experiment_id="exp-trend")

    assert trend.has_trend is True
    # Verify sorted chronologically
    assert trend.run_ids == ["run-1", "run-2", "run-3"]
    assert trend.timestamps == [
        "2026-11-01T10:00:00Z",
        "2026-11-01T11:00:00Z",
        "2026-11-01T12:00:00Z",
    ]

    # Verify transitions: r3 (index 2) has fingerprint fp-2, differing from fp-1
    assert trend.configuration_transitions == [2]
    assert trend.is_repeated_run_trend is False

    # Verify metric series
    assert "exact_match_pass_rate" in trend.metric_series
    assert trend.metric_series["exact_match_pass_rate"] == [1.0, 0.0, 1.0]
    assert trend.metric_series["exact_match_mean_score"] == [1.0, 0.5, 0.8]
