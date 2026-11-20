"""
Repeated-run consistency (RQ4): how much do identical executions vary?

A *repeat group* is a set of runs that are true repeats of each other: the
same ``experiment_id``, the same ``config_fingerprint`` (dataset version,
model configuration and evaluator configurations) and the same
``adapter_id`` (the adapter that actually produced the responses; it is not
part of the fingerprint, so two runs of one experiment against different
adapters are a model comparison, never a repeat). Runs whose ``adapter_id``
was not recorded (``None``, persisted before the field existed) form their
own group and are flagged ``adapter_recorded=False``, because it cannot be
confirmed that they used one adapter.

Only runs with a ``result`` are considered, and only groups with at least two
such runs are reported. Within a group, runs are ordered by
``repetition_index`` and then ``started_at``.

Metric definitions (every rate carries its numerator and denominator):

- **Compared cases**: test case ids that have a model response in *every*
  run of the group. A case missing from a run, or whose generation failed
  in any run (no ``model_response``), is excluded and counted in
  ``excluded_cases``; nothing is imputed for it.
- **Identical-output rate** = (compared cases whose ``output_text`` is the
  exact same string in every run) / (compared cases). Exact string equality:
  no case folding, whitespace or punctuation normalization, so a change of
  capitalisation or a trailing full stop counts as a different output.
- **Verdict stability, per evaluator** = (cases where the evaluator returned
  ``SUCCESS`` with a non-``None`` ``passed`` value in every run, and the same
  ``passed`` value in every run) / (cases where it returned such a verdict in
  every run). A case is excluded from that evaluator's denominator, and
  counted separately, when in any run the evaluator result is absent
  (``excluded_missing``), its status is not ``SUCCESS`` (``excluded_not_success``),
  or it is ``SUCCESS`` with ``passed=None``, e.g. a faithfulness judge's
  ``undetermined`` label (``excluded_no_verdict``). Errors are never counted
  as a stable or unstable verdict.
- **Case-level pass stability** = (compared cases whose
  ``TestCaseResult.passed`` is the same in every run) / (compared cases).
  ``TestCaseResult.passed`` is a strict bool (an evaluator error makes it
  ``False``), so an evaluator error in one run of an otherwise passing case
  shows up here as a flip. The per-evaluator figures above are the ones that
  separate evaluator errors from verdict changes.

A high identical-output rate at temperature 0 is evidence of execution
determinism for that model/server; it does not imply correctness, and a
small number of compared cases gives a wide uncertainty range.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from llm_reliability.evaluation.models import EvaluationStatus, TestCaseResult
from llm_reliability.experiments.models import ExperimentRun

# Upper bound on how many test case ids each "which cases changed" list holds.
# The matching ``*_count`` field always reports the full total.
MAX_LISTED_IDS = 50


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator > 0 else None


@dataclass
class EvaluatorStability:
    """Verdict stability of one evaluator across the runs of a repeat group."""

    evaluator_name: str
    stable_cases: int
    compared_cases: int
    stability_rate: float | None
    excluded_missing: int
    excluded_not_success: int
    excluded_no_verdict: int
    unstable_count: int
    unstable_test_case_ids: list[str] = field(default_factory=list)

    @property
    def excluded_cases(self) -> int:
        return self.excluded_missing + self.excluded_not_success + self.excluded_no_verdict

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator_name": self.evaluator_name,
            "stable_cases": self.stable_cases,
            "compared_cases": self.compared_cases,
            "stability_rate": self.stability_rate,
            "excluded_cases": self.excluded_cases,
            "excluded_missing": self.excluded_missing,
            "excluded_not_success": self.excluded_not_success,
            "excluded_no_verdict": self.excluded_no_verdict,
            "unstable_count": self.unstable_count,
            "unstable_test_case_ids": list(self.unstable_test_case_ids),
        }


@dataclass
class RepeatGroupConsistency:
    """Consistency of one group of true repeat runs (see module docstring)."""

    experiment_id: str
    experiment_name: str | None
    config_fingerprint: str
    adapter_id: str | None
    adapter_recorded: bool
    run_ids: list[str]
    temperature: float | None
    model_id: str
    simulated_components: list[str]
    total_test_cases: int
    compared_cases: int
    excluded_cases: int
    identical_output_cases: int
    identical_output_rate: float | None
    output_changed_count: int
    output_changed_test_case_ids: list[str]
    case_pass_stable_cases: int
    case_pass_compared_cases: int
    case_pass_stability_rate: float | None
    case_pass_changed_count: int
    case_pass_changed_test_case_ids: list[str]
    evaluators: list[EvaluatorStability] = field(default_factory=list)

    @property
    def num_runs(self) -> int:
        return len(self.run_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "config_fingerprint": self.config_fingerprint,
            "adapter_id": self.adapter_id,
            "adapter_recorded": self.adapter_recorded,
            "num_runs": self.num_runs,
            "run_ids": list(self.run_ids),
            "temperature": self.temperature,
            "model_id": self.model_id,
            "simulated_components": list(self.simulated_components),
            "total_test_cases": self.total_test_cases,
            "compared_cases": self.compared_cases,
            "excluded_cases": self.excluded_cases,
            "identical_output_cases": self.identical_output_cases,
            "identical_output_rate": self.identical_output_rate,
            "output_changed_count": self.output_changed_count,
            "output_changed_test_case_ids": list(self.output_changed_test_case_ids),
            "case_pass_stable_cases": self.case_pass_stable_cases,
            "case_pass_compared_cases": self.case_pass_compared_cases,
            "case_pass_stability_rate": self.case_pass_stability_rate,
            "case_pass_changed_count": self.case_pass_changed_count,
            "case_pass_changed_test_case_ids": list(self.case_pass_changed_test_case_ids),
            "evaluators": [e.to_dict() for e in self.evaluators],
        }


def _run_order_key(run: ExperimentRun) -> tuple[int, str, str]:
    return (run.repetition_index, run.started_at, run.run_id)


def group_repeat_runs(
    runs: Iterable[ExperimentRun],
) -> list[list[ExperimentRun]]:
    """Group runs into true repeats; keep only groups with >= 2 runs that have results.

    Key: (experiment_id, config_fingerprint, adapter_id). ``adapter_id=None``
    is its own key value, so unrecorded-adapter runs never merge with runs
    whose adapter is known. Groups are returned in a stable order
    (experiment id, then fingerprint, then adapter id) with runs sorted by
    ``repetition_index`` then ``started_at``.
    """
    groups: dict[tuple[str, str, str | None], list[ExperimentRun]] = {}
    for run in runs:
        if run.result is None:
            continue
        key = (run.experiment_id, run.config_fingerprint, run.adapter_id)
        groups.setdefault(key, []).append(run)
    ordered_keys = sorted(groups, key=lambda k: (k[0], k[1], k[2] is None, k[2] or ""))
    return [sorted(groups[k], key=_run_order_key) for k in ordered_keys if len(groups[k]) >= 2]


def _evaluator_stability(
    evaluator_name: str,
    case_ids: list[str],
    by_run: list[dict[str, TestCaseResult]],
) -> EvaluatorStability:
    stable = 0
    compared = 0
    missing = 0
    not_success = 0
    no_verdict = 0
    unstable_ids: list[str] = []
    for case_id in case_ids:
        found = [
            next(
                (
                    r
                    for r in results[case_id].evaluation_results
                    if r.evaluator_name == evaluator_name
                ),
                None,
            )
            for results in by_run
        ]
        # Exclusion precedence when runs differ: missing > not SUCCESS > no verdict.
        if any(res is None for res in found):
            missing += 1
            continue
        present = [res for res in found if res is not None]
        if any(res.status != EvaluationStatus.SUCCESS for res in present):
            not_success += 1
            continue
        if any(res.passed is None for res in present):
            no_verdict += 1
            continue
        compared += 1
        if len({res.passed for res in present}) == 1:
            stable += 1
        else:
            unstable_ids.append(case_id)
    return EvaluatorStability(
        evaluator_name=evaluator_name,
        stable_cases=stable,
        compared_cases=compared,
        stability_rate=_rate(stable, compared),
        excluded_missing=missing,
        excluded_not_success=not_success,
        excluded_no_verdict=no_verdict,
        unstable_count=len(unstable_ids),
        unstable_test_case_ids=unstable_ids[:MAX_LISTED_IDS],
    )


def compute_group_consistency(
    runs: list[ExperimentRun], experiment_name: str | None = None
) -> RepeatGroupConsistency:
    """Compute consistency metrics for one repeat group (see module docstring).

    ``runs`` must already be a valid repeat group, as produced by
    :func:`group_repeat_runs`: two or more runs with results sharing
    experiment id, fingerprint and adapter id.
    """
    if len(runs) < 2:
        raise ValueError("A repeat group needs at least two runs.")
    first = runs[0]
    keys = {(r.experiment_id, r.config_fingerprint, r.adapter_id) for r in runs}
    if len(keys) != 1 or any(r.result is None for r in runs):
        raise ValueError(
            "Runs in a repeat group must all have results and share experiment_id, "
            "config_fingerprint and adapter_id."
        )

    by_run: list[dict[str, TestCaseResult]] = [
        {tc.test_case_id: tc for tc in r.result.test_case_results}  # type: ignore[union-attr]
        for r in runs
    ]
    all_ids: dict[str, None] = {}
    for results in by_run:
        for case_id in results:
            all_ids.setdefault(case_id, None)
    compared_ids = [
        case_id
        for case_id in all_ids
        if all(
            case_id in results and results[case_id].model_response is not None for results in by_run
        )
    ]

    output_changed: list[str] = []
    pass_changed: list[str] = []
    for case_id in compared_ids:
        outputs = {results[case_id].model_response.output_text for results in by_run}  # type: ignore[union-attr]
        if len(outputs) > 1:
            output_changed.append(case_id)
        if len({results[case_id].passed for results in by_run}) > 1:
            pass_changed.append(case_id)

    evaluator_names: dict[str, None] = {}
    for r in runs:
        for name in r.result.evaluator_names:  # type: ignore[union-attr]
            evaluator_names.setdefault(name, None)
        for tc in r.result.test_case_results:  # type: ignore[union-attr]
            for er in tc.evaluation_results:
                evaluator_names.setdefault(er.evaluator_name, None)

    simulated: dict[str, None] = {}
    for r in runs:
        for component in r.simulated_components:
            simulated.setdefault(component, None)

    n = len(compared_ids)
    identical = n - len(output_changed)
    pass_stable = n - len(pass_changed)
    return RepeatGroupConsistency(
        experiment_id=first.experiment_id,
        experiment_name=experiment_name,
        config_fingerprint=first.config_fingerprint,
        adapter_id=first.adapter_id,
        adapter_recorded=first.adapter_id is not None,
        run_ids=[r.run_id for r in runs],
        temperature=first.model_config.temperature,
        model_id=first.model_config.model_id,
        simulated_components=list(simulated),
        total_test_cases=len(all_ids),
        compared_cases=n,
        excluded_cases=len(all_ids) - n,
        identical_output_cases=identical,
        identical_output_rate=_rate(identical, n),
        output_changed_count=len(output_changed),
        output_changed_test_case_ids=output_changed[:MAX_LISTED_IDS],
        case_pass_stable_cases=pass_stable,
        case_pass_compared_cases=n,
        case_pass_stability_rate=_rate(pass_stable, n),
        case_pass_changed_count=len(pass_changed),
        case_pass_changed_test_case_ids=pass_changed[:MAX_LISTED_IDS],
        evaluators=[
            _evaluator_stability(name, compared_ids, by_run) for name in sorted(evaluator_names)
        ],
    )


def compute_repeat_consistency(
    runs: Iterable[ExperimentRun],
    experiment_names: dict[str, str] | None = None,
) -> list[RepeatGroupConsistency]:
    """Group ``runs`` into true repeats and compute consistency for each group."""
    names = experiment_names or {}
    return [
        compute_group_consistency(group, experiment_name=names.get(group[0].experiment_id))
        for group in group_repeat_runs(runs)
    ]
