"""
Tests for repeated-run consistency (RQ4) with hand-built runs.
"""

from __future__ import annotations

import pytest

from llm_reliability.analytics.consistency import (
    MAX_LISTED_IDS,
    compute_group_consistency,
    compute_repeat_consistency,
    group_repeat_runs,
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

CASE_IDS = ["tc-1", "tc-2", "tc-3"]


def _definition(
    experiment_id: str = "exp-rep", temperature: float | None = 0.0
) -> ExperimentDefinition:
    ds = Dataset(
        name="rep",
        test_cases=[TestCase(id=c, input=f"Q {c}", reference_answer="A") for c in CASE_IDS],
    )
    return ExperimentDefinition(
        experiment_id=experiment_id,
        name="Repeat test",
        dataset_version=DatasetVersion(dataset=ds),
        model_config=ModelConfig(model_id="real-model", temperature=temperature),
        evaluator_configs=[
            EvaluatorConfig(evaluator_name="exact_match"),
            EvaluatorConfig(evaluator_name="llm_judge"),
        ],
    )


def _eval(
    name: str,
    case_id: str,
    passed: bool | None,
    status: EvaluationStatus = EvaluationStatus.SUCCESS,
) -> EvaluationResult:
    if status != EvaluationStatus.SUCCESS:
        return EvaluationResult(
            evaluator_name=name, test_case_id=case_id, status=status, error="boom"
        )
    return EvaluationResult(
        evaluator_name=name, test_case_id=case_id, status=status, passed=passed, criterion="c"
    )


def _case(
    case_id: str,
    output: str | None,
    em: bool | None = True,
    judge: bool | None = True,
    judge_status: EvaluationStatus = EvaluationStatus.SUCCESS,
    provider_metadata: dict | None = None,
) -> TestCaseResult:
    if output is None:
        return TestCaseResult(
            test_case_id=case_id,
            input_text=f"Q {case_id}",
            reference_answer="A",
            model_response=None,
            evaluation_results=[],
            execution_error="generation failed",
        )
    return TestCaseResult(
        test_case_id=case_id,
        input_text=f"Q {case_id}",
        reference_answer="A",
        model_response=ModelResponse(
            test_case_id=case_id,
            output_text=output,
            model_id="real-model",
            provider_metadata=provider_metadata or {},
        ),
        evaluation_results=[
            _eval("exact_match", case_id, em),
            _eval("llm_judge", case_id, judge, judge_status),
        ],
    )


def _run(
    run_id: str,
    cases: list[TestCaseResult],
    adapter_id: str | None = "adapter-a",
    definition: ExperimentDefinition | None = None,
    repetition_index: int = 0,
    with_result: bool = True,
) -> ExperimentRun:
    d = definition or _definition()
    return ExperimentRun(
        run_id=run_id,
        experiment_id=d.experiment_id,
        status=RunStatus.SUCCEEDED,
        started_at=f"2026-01-01T00:00:0{repetition_index}+00:00",
        finished_at="2026-01-01T00:01:00+00:00",
        dataset_version=d.dataset_version,
        model_config=d.model_config,
        evaluator_configs=d.evaluator_configs,
        config_fingerprint=d.config_fingerprint,
        result=(
            RunResult(
                dataset_name="rep",
                model_id="real-model",
                evaluator_names=["exact_match", "llm_judge"],
                test_case_results=cases,
            )
            if with_result
            else None
        ),
        adapter_id=adapter_id,
        repetition_index=repetition_index,
    )


def _same_cases() -> list[TestCaseResult]:
    return [
        _case("tc-1", "Paris"),
        _case("tc-2", "Rome", em=False, judge=True),
        _case("tc-3", "Oslo"),
    ]


def _evaluator(group, name):
    return next(e for e in group.evaluators if e.evaluator_name == name)


def test_identical_repeats_are_fully_consistent() -> None:
    runs = [
        _run("r1", _same_cases(), repetition_index=1),
        _run("r2", _same_cases(), repetition_index=2),
    ]
    [group] = compute_repeat_consistency(runs, experiment_names={"exp-rep": "Repeat test"})

    assert group.run_ids == ["r1", "r2"]
    assert group.experiment_name == "Repeat test"
    assert group.temperature == 0.0
    assert group.adapter_recorded is True
    assert group.simulated_components == []
    assert (group.identical_output_cases, group.compared_cases) == (3, 3)
    assert group.identical_output_rate == 1.0
    assert group.output_changed_count == 0
    assert (group.case_pass_stable_cases, group.case_pass_compared_cases) == (3, 3)
    for name in ("exact_match", "llm_judge"):
        ev = _evaluator(group, name)
        assert (ev.stable_cases, ev.compared_cases, ev.excluded_cases) == (3, 3, 0)
        assert ev.stability_rate == 1.0


def test_differing_outputs_are_counted_with_exact_string_equality() -> None:
    r1 = _run("r1", _same_cases(), repetition_index=1)
    # tc-1 differs only by a trailing full stop: exact equality counts it as changed.
    # tc-3 changes wording and the judge flips its verdict.
    r2 = _run(
        "r2",
        [
            _case("tc-1", "Paris."),
            _case("tc-2", "Rome", em=False, judge=True),
            _case("tc-3", "It is Oslo", em=False, judge=False),
        ],
        repetition_index=2,
    )
    [group] = compute_repeat_consistency([r2, r1])

    assert group.run_ids == ["r1", "r2"]  # ordered by repetition_index, not input order
    assert (group.identical_output_cases, group.compared_cases) == (1, 3)
    assert group.identical_output_rate == pytest.approx(1 / 3)
    assert group.output_changed_count == 2
    assert group.output_changed_test_case_ids == ["tc-1", "tc-3"]
    judge = _evaluator(group, "llm_judge")
    assert (judge.stable_cases, judge.compared_cases) == (2, 3)
    assert judge.unstable_test_case_ids == ["tc-3"]
    em = _evaluator(group, "exact_match")
    assert (em.stable_cases, em.compared_cases) == (2, 3)
    # Case-level: tc-2 fails in both runs (exact match False) -> stable; tc-3 flips.
    assert (group.case_pass_stable_cases, group.case_pass_compared_cases) == (2, 3)
    assert group.case_pass_changed_test_case_ids == ["tc-3"]


def test_evaluator_errors_and_no_verdict_are_excluded_not_counted_as_flips() -> None:
    r1 = _run("r1", _same_cases(), repetition_index=1)
    r2 = _run(
        "r2",
        [
            _case("tc-1", "Paris", judge_status=EvaluationStatus.EXECUTION_ERROR),
            _case("tc-2", "Rome", em=False, judge=None),  # SUCCESS but no verdict
            _case("tc-3", "Oslo"),
        ],
        repetition_index=2,
    )
    [group] = compute_repeat_consistency([r1, r2])

    judge = _evaluator(group, "llm_judge")
    assert judge.compared_cases == 1
    assert judge.stable_cases == 1
    assert judge.excluded_not_success == 1
    assert judge.excluded_no_verdict == 1
    assert judge.excluded_missing == 0
    assert judge.excluded_cases == 2
    assert judge.unstable_count == 0
    # The other evaluator is unaffected.
    em = _evaluator(group, "exact_match")
    assert (em.stable_cases, em.compared_cases, em.excluded_cases) == (3, 3, 0)
    # TestCaseResult.passed is a strict bool: an evaluator error turns tc-1 into a
    # case-level flip (documented limitation), while outputs stay identical.
    assert group.identical_output_rate == 1.0
    assert group.case_pass_changed_test_case_ids == ["tc-1"]


def test_cases_without_a_response_in_every_run_are_excluded() -> None:
    r1 = _run("r1", _same_cases(), repetition_index=1)
    r2 = _run(
        "r2",
        [_case("tc-1", "Paris"), _case("tc-2", None)],  # tc-2 crashed, tc-3 absent
        repetition_index=2,
    )
    [group] = compute_repeat_consistency([r1, r2])

    assert group.total_test_cases == 3
    assert group.compared_cases == 1
    assert group.excluded_cases == 2
    assert (group.identical_output_cases, group.compared_cases) == (1, 1)
    assert _evaluator(group, "llm_judge").compared_cases == 1


def test_missing_evaluator_result_is_excluded() -> None:
    r1 = _run("r1", _same_cases(), repetition_index=1)
    cases = _same_cases()
    cases[0].evaluation_results = [
        r for r in cases[0].evaluation_results if r.evaluator_name != "llm_judge"
    ]
    r2 = _run("r2", cases, repetition_index=2)
    [group] = compute_repeat_consistency([r1, r2])

    judge = _evaluator(group, "llm_judge")
    assert (judge.excluded_missing, judge.compared_cases) == (1, 2)


def test_different_adapters_are_not_grouped() -> None:
    runs = [
        _run("r1", _same_cases(), adapter_id="adapter-a"),
        _run("r2", _same_cases(), adapter_id="adapter-b"),
    ]
    assert compute_repeat_consistency(runs) == []


def test_unrecorded_adapter_forms_its_own_flagged_group() -> None:
    runs = [
        _run("known", _same_cases(), adapter_id="adapter-a"),
        _run("legacy-1", _same_cases(), adapter_id=None, repetition_index=1),
        _run("legacy-2", _same_cases(), adapter_id=None, repetition_index=2),
    ]
    [group] = compute_repeat_consistency(runs)
    assert group.run_ids == ["legacy-1", "legacy-2"]
    assert group.adapter_id is None
    assert group.adapter_recorded is False


def test_single_runs_and_runs_without_results_are_ignored() -> None:
    other = _definition(experiment_id="exp-other")
    runs = [
        _run("solo", _same_cases()),
        _run("no-result", [], with_result=False),
        _run("other-1", _same_cases(), definition=other),
    ]
    assert group_repeat_runs(runs) == []
    assert compute_repeat_consistency(runs) == []


def test_different_fingerprints_are_not_grouped() -> None:
    hot = _definition(temperature=0.8)
    runs = [_run("cold", _same_cases()), _run("hot", _same_cases(), definition=hot)]
    assert hot.config_fingerprint != _definition().config_fingerprint
    assert compute_repeat_consistency(runs) == []


def test_simulated_components_are_unioned_across_runs() -> None:
    mock_cases = [_case(c, "x", provider_metadata={"adapter": "mock"}) for c in CASE_IDS]
    runs = [
        _run("r1", _same_cases(), repetition_index=1),
        _run("r2", mock_cases, repetition_index=2),
    ]
    [group] = compute_repeat_consistency(runs)
    assert group.simulated_components == ["generation: deterministic mock adapter (canned answers)"]


def test_listed_ids_are_capped_but_totals_are_complete() -> None:
    n = MAX_LISTED_IDS + 5
    ids = [f"tc-{i}" for i in range(n)]
    r1 = _run("r1", [_case(i, "a") for i in ids], repetition_index=1)
    r2 = _run("r2", [_case(i, "b") for i in ids], repetition_index=2)
    group = compute_group_consistency([r1, r2])
    assert group.output_changed_count == n
    assert len(group.output_changed_test_case_ids) == MAX_LISTED_IDS
    assert group.identical_output_rate == 0.0


def test_compute_group_consistency_rejects_non_repeats() -> None:
    with pytest.raises(ValueError):
        compute_group_consistency([_run("r1", _same_cases())])
    with pytest.raises(ValueError):
        compute_group_consistency(
            [_run("r1", _same_cases(), adapter_id="a"), _run("r2", _same_cases(), adapter_id="b")]
        )
