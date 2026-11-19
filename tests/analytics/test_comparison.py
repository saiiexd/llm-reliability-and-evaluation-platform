"""
Tests for run and experiment comparison logic.
"""

from __future__ import annotations

from llm_reliability.analytics.comparison import assess_compatibility, compare_runs
from llm_reliability.analytics.models import ComparisonCompatibility
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


def _make_run(
    run_id: str,
    dataset_name: str,
    input_text: str,
    fingerprint: str,
    temp: float = 0.0,
    em_score: float = 1.0,
) -> ExperimentRun:
    ds = Dataset(
        name=dataset_name,
        test_cases=[TestCase(id="tc-1", input=input_text, reference_answer="Paris")],
    )
    defn = ExperimentDefinition(
        experiment_id=f"exp-{dataset_name}",
        name="Comp Exp",
        dataset_version=DatasetVersion(dataset=ds),
        model_config=ModelConfig(model_id="mock-model", temperature=temp),
        evaluator_configs=[EvaluatorConfig(evaluator_name="exact_match")],
    )
    tc = TestCaseResult(
        test_case_id="tc-1",
        input_text=input_text,
        reference_answer="Paris",
        model_response=ModelResponse(
            test_case_id="tc-1", output_text="Paris", model_id="mock-model"
        ),
        evaluation_results=[
            EvaluationResult(
                evaluator_name="exact_match",
                test_case_id="tc-1",
                status=EvaluationStatus.SUCCESS,
                score=em_score,
                passed=(em_score > 0.5),
                criterion="correctness",
            )
        ],
        duration_seconds=0.05,
    )
    return ExperimentRun(
        run_id=run_id,
        experiment_id=f"exp-{dataset_name}",
        status=RunStatus.SUCCEEDED,
        started_at="2026-11-01T10:00:00Z",
        finished_at="2026-11-01T10:00:01Z",
        dataset_version=defn.dataset_version,
        model_config=defn.model_config,
        evaluator_configs=defn.evaluator_configs,
        config_fingerprint=fingerprint,
        result=RunResult(
            dataset_name=dataset_name,
            model_id="mock-model",
            evaluator_names=["exact_match"],
            test_case_results=[tc],
        ),
    )


def test_assess_compatibility_identical() -> None:
    r1 = _make_run("run-1", "ds-v1", "Q1", "fp-aaa", temp=0.0)
    r2 = _make_run("run-2", "ds-v1", "Q1", "fp-aaa", temp=0.0)

    compat, reasons, diffs = assess_compatibility([r1, r2])
    assert compat == ComparisonCompatibility.IDENTICAL_CONFIGURATION
    assert "identical configuration fingerprint" in reasons[0].lower()


def test_assess_compatibility_partially_different() -> None:
    r1 = _make_run("run-1", "ds-v1", "Q1", "fp-aaa", temp=0.0)
    r2 = _make_run("run-2", "ds-v1", "Q1", "fp-bbb", temp=0.7)

    compat, reasons, diffs = assess_compatibility([r1, r2])
    assert compat == ComparisonCompatibility.PARTIALLY_DIFFERENT_CONFIGURATION
    assert "controlled comparative analysis" in reasons[0].lower()
    assert diffs["model_configs"]["run-1"]["temperature"] == 0.0
    assert diffs["model_configs"]["run-2"]["temperature"] == 0.7


def test_assess_compatibility_materially_incompatible() -> None:
    # Different input text produces different version_id
    r1 = _make_run("run-1", "ds-v1", "Q1", "fp-aaa")
    r2 = _make_run("run-2", "ds-v2", "Q2-different", "fp-bbb")

    compat, reasons, diffs = assess_compatibility([r1, r2])
    assert compat == ComparisonCompatibility.MATERIALLY_INCOMPATIBLE
    assert "differing dataset versions" in reasons[0].lower()


def test_compare_runs() -> None:
    r1 = _make_run("run-1", "ds-v1", "Q1", "fp-aaa", temp=0.0, em_score=1.0)
    r2 = _make_run("run-2", "ds-v1", "Q1", "fp-bbb", temp=0.7, em_score=0.0)

    comparison = compare_runs([r1, r2])
    assert comparison.source_run_ids == ["run-1", "run-2"]
    assert comparison.compatibility == ComparisonCompatibility.PARTIALLY_DIFFERENT_CONFIGURATION

    em_comp = comparison.evaluation_comparisons.get("exact_match")
    assert em_comp is not None
    assert em_comp["pass_rates"]["run-1"] == 1.0
    assert em_comp["pass_rates"]["run-2"] == 0.0
    assert em_comp["numerators"]["run-1"] == 1
    assert em_comp["numerators"]["run-2"] == 0
    assert em_comp["denominators"]["run-1"] == 1
    assert em_comp["denominators"]["run-2"] == 1
