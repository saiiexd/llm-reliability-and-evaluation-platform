"""
Aggregation functions for evaluation results, failure distributions, retrieval, and performance.

All aggregation logic preserves numerators, denominators, and underlying counts.
Never collapses distinct evaluator methodologies into a single synthetic score.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from llm_reliability.analytics.models import (
    EvaluationAggregate,
    ExecutionPerformanceSummary,
    FailureCategoryCount,
    FailureDistribution,
    RetrievalSummary,
    RobustnessSummary,
    SecondaryFindingCount,
)
from llm_reliability.diagnostics.diagnosis import ReliabilityDiagnostic
from llm_reliability.diagnostics.robustness import RobustnessResult
from llm_reliability.diagnostics.taxonomy import DiagnosticStatus, FailureCategory
from llm_reliability.evaluation.models import EvaluationStatus
from llm_reliability.experiments.models import ExperimentRun
from llm_reliability.rag.evaluators import RetrievalEvaluationResult, RetrievalEvaluationStatus


def aggregate_evaluation(run: ExperimentRun) -> dict[str, EvaluationAggregate]:
    """Aggregate evaluation outcomes per evaluator for one experiment run.

    Keeps each evaluator and criterion separate. Preserves numerators and
    denominators for all pass rates.
    """
    if run.result is None:
        return {}

    test_cases = run.result.test_case_results
    total_cases = len(test_cases)
    by_evaluator: dict[str, list[Any]] = {}

    for tc in test_cases:
        for res in tc.evaluation_results:
            by_evaluator.setdefault(res.evaluator_name, []).append(res)

    aggregates: dict[str, EvaluationAggregate] = {}

    for name, results in by_evaluator.items():
        criterion = results[0].criterion if results else None
        num_evaluated = len(results)
        num_success = sum(1 for r in results if r.status == EvaluationStatus.SUCCESS)
        num_skipped = sum(1 for r in results if r.status == EvaluationStatus.SKIPPED)
        num_invalid = sum(1 for r in results if r.status == EvaluationStatus.INVALID_CONFIGURATION)
        num_exec_err = sum(1 for r in results if r.status == EvaluationStatus.EXECUTION_ERROR)
        num_val_err = sum(
            1 for r in results if r.status == EvaluationStatus.OUTPUT_VALIDATION_ERROR
        )

        num_passed = sum(1 for r in results if r.passed is True)
        num_failed = sum(1 for r in results if r.passed is False)

        decided_count = num_passed + num_failed
        pass_rate = (num_passed / decided_count) if decided_count > 0 else None

        numeric_scores = [r.score for r in results if r.score is not None]
        score_count = len(numeric_scores)
        score_mean = (sum(numeric_scores) / score_count) if score_count > 0 else None
        score_min = min(numeric_scores) if score_count > 0 else None
        score_max = max(numeric_scores) if score_count > 0 else None

        criterion_outcomes: dict[str, int] = {}
        for r in results:
            if r.label:
                criterion_outcomes[r.label] = criterion_outcomes.get(r.label, 0) + 1

        aggregates[name] = EvaluationAggregate(
            evaluator_name=name,
            criterion=criterion,
            total_cases=total_cases,
            num_evaluated=num_evaluated,
            num_success=num_success,
            num_skipped=num_skipped,
            num_invalid_config=num_invalid,
            num_execution_error=num_exec_err,
            num_output_validation_error=num_val_err,
            num_passed=num_passed,
            num_failed=num_failed,
            pass_rate=pass_rate,
            pass_rate_numerator=num_passed,
            pass_rate_denominator=decided_count,
            score_count=score_count,
            score_mean=score_mean,
            score_min=score_min,
            score_max=score_max,
            criterion_outcomes=criterion_outcomes,
        )

    return aggregates


def aggregate_failures(
    diagnostics: dict[str, ReliabilityDiagnostic] | list[ReliabilityDiagnostic],
    source_run_ids: list[str] | str | None = None,
) -> FailureDistribution:
    """Summarize primary failure category distributions and secondary finding frequencies."""
    if isinstance(diagnostics, list):
        diag_list = diagnostics
    else:
        diag_list = list(diagnostics.values())

    total_cases = len(diag_list)
    num_determined = sum(1 for d in diag_list if d.status == DiagnosticStatus.DETERMINED)
    num_undetermined = total_cases - num_determined

    # Primary category frequencies
    primary_counts: dict[str, int] = {cat.value: 0 for cat in FailureCategory}
    for d in diag_list:
        if d.failure_category:
            primary_counts[d.failure_category.value] = (
                primary_counts.get(d.failure_category.value, 0) + 1
            )

    primary_categories: dict[str, FailureCategoryCount] = {}
    for cat_name, count in primary_counts.items():
        proportion = (count / total_cases) if total_cases > 0 else None
        primary_categories[cat_name] = FailureCategoryCount(
            category=FailureCategory(cat_name),
            count=count,
            proportion=proportion,
            denominator=total_cases,
        )

    # Secondary finding frequencies
    secondary_counts: dict[str, int] = {}
    for d in diag_list:
        for finding in d.secondary_findings:
            key = finding.split(":")[0].strip() if ":" in finding else finding.strip()
            secondary_counts[key] = secondary_counts.get(key, 0) + 1

    secondary_findings: dict[str, SecondaryFindingCount] = {}
    for finding_name, count in secondary_counts.items():
        freq = (count / total_cases) if total_cases > 0 else None
        secondary_findings[finding_name] = SecondaryFindingCount(
            finding_name=finding_name,
            count=count,
            frequency=freq,
            denominator=total_cases,
        )

    if isinstance(source_run_ids, str):
        run_ids = [source_run_ids]
    elif source_run_ids is not None:
        run_ids = list(source_run_ids)
    else:
        run_ids = []

    return FailureDistribution(
        total_cases=total_cases,
        num_determined=num_determined,
        num_undetermined=num_undetermined,
        primary_categories=primary_categories,
        secondary_findings=secondary_findings,
        source_run_ids=run_ids,
    )


def aggregate_retrieval(
    retrieval_results: dict[str, list[RetrievalEvaluationResult]] | list[RetrievalEvaluationResult],
) -> list[RetrievalSummary]:
    """Summarize retrieval metrics preserving metric names, K values, and ground-truth counts."""
    grouped: dict[tuple[str, int | None], list[RetrievalEvaluationResult]] = {}
    if isinstance(retrieval_results, dict):
        total_cases = len(retrieval_results)
        for case_results in retrieval_results.values():
            for res in case_results:
                k = res.evaluator_config.get("k") if res.evaluator_config else None
                key = (res.metric_name, k)
                grouped.setdefault(key, []).append(res)
    else:
        distinct_cases = {res.test_case_id for res in retrieval_results}
        total_cases = len(distinct_cases) if distinct_cases else len(retrieval_results)
        for res in retrieval_results:
            k = res.evaluator_config.get("k") if res.evaluator_config else None
            key = (res.metric_name, k)
            grouped.setdefault(key, []).append(res)

    summaries: list[RetrievalSummary] = []
    sorted_items = sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1] or 0))
    for (metric_name, k), results in sorted_items:
        evaluated = [
            r
            for r in results
            if r.status == RetrievalEvaluationStatus.SUCCESS and r.score is not None
        ]
        skipped = [r for r in results if r.status == RetrievalEvaluationStatus.SKIPPED]

        scores = [r.score for r in evaluated if r.score is not None]
        mean_score = (sum(scores) / len(scores)) if scores else None
        min_score = min(scores) if scores else None
        max_score = max(scores) if scores else None

        summaries.append(
            RetrievalSummary(
                metric_name=metric_name,
                k=k,
                total_cases=total_cases,
                evaluated_cases=len(evaluated),
                skipped_cases=len(skipped),
                mean_score=mean_score,
                min_score=min_score,
                max_score=max_score,
            )
        )

    return summaries


def aggregate_robustness(robustness_results: list[RobustnessResult]) -> RobustnessSummary:
    """Summarize robustness consistency and transition dynamics across perturbation types."""
    total_pairs = len(robustness_results)
    num_unchanged = sum(1 for r in robustness_results if not r.outcome_changed)
    num_changed = total_pairs - num_unchanged
    consistency_rate = (num_unchanged / total_pairs) if total_pairs > 0 else None

    by_perturbation: dict[str, dict[str, Any]] = {}
    transition_counts: dict[str, int] = {}

    for r in robustness_results:
        p_type = r.perturbation_type.value
        if p_type not in by_perturbation:
            by_perturbation[p_type] = {"total": 0, "unchanged": 0, "changed": 0, "rate": None}
        by_perturbation[p_type]["total"] += 1
        if not r.outcome_changed:
            by_perturbation[p_type]["unchanged"] += 1
        else:
            by_perturbation[p_type]["changed"] += 1

        if r.failure_category_transition:
            trans = r.failure_category_transition
            transition_counts[trans] = transition_counts.get(trans, 0) + 1

    for stats in by_perturbation.values():
        if stats["total"] > 0:
            stats["rate"] = stats["unchanged"] / stats["total"]

    return RobustnessSummary(
        total_pairs=total_pairs,
        num_unchanged=num_unchanged,
        num_changed=num_changed,
        consistency_rate=consistency_rate,
        by_perturbation=by_perturbation,
        transition_counts=transition_counts,
    )


def measure_performance(run: ExperimentRun) -> ExecutionPerformanceSummary:
    """Calculate measured execution timing and explicitly report telemetry availability."""
    notes: list[str] = []

    # Wall-clock duration from ISO timestamps
    wall_clock: float | None = None
    try:
        start_dt = datetime.fromisoformat(run.started_at)
        finish_dt = datetime.fromisoformat(run.finished_at)
        wall_clock = max(0.0, (finish_dt - start_dt).total_seconds())
        notes.append("Wall-clock run duration derived from UTC lifecycle timestamps.")
    except Exception:
        notes.append("Could not parse start or finish timestamps.")

    per_case_mean: float | None = None
    evaluator_mean: float | None = None
    tokens_available = False

    if run.result and run.result.test_case_results:
        case_durations = [
            tc.duration_seconds
            for tc in run.result.test_case_results
            if tc.duration_seconds is not None
        ]
        if case_durations:
            per_case_mean = sum(case_durations) / len(case_durations)
            notes.append("Per-test-case execution durations measured via monotonic clock.")
        else:
            notes.append("Per-test-case duration was not instrumented for this run.")

        eval_durations = [
            res.duration_seconds
            for tc in run.result.test_case_results
            for res in tc.evaluation_results
            if res.duration_seconds is not None
        ]
        if eval_durations:
            evaluator_mean = sum(eval_durations) / len(eval_durations)
            notes.append("Per-evaluator durations measured via monotonic clock.")

        tokens_present = any(
            tc.model_response is not None
            and (
                tc.model_response.prompt_tokens is not None
                or tc.model_response.completion_tokens is not None
            )
            for tc in run.result.test_case_results
        )
        tokens_available = tokens_present

        if tokens_present:
            prompt_tokens_total = sum(
                tc.model_response.prompt_tokens
                for tc in run.result.test_case_results
                if tc.model_response and tc.model_response.prompt_tokens is not None
            )
            completion_tokens_total = sum(
                tc.model_response.completion_tokens
                for tc in run.result.test_case_results
                if tc.model_response and tc.model_response.completion_tokens is not None
            )
            total_tokens = sum(
                tc.model_response.total_tokens
                for tc in run.result.test_case_results
                if tc.model_response and tc.model_response.total_tokens is not None
            )
            notes.append(
                f"Token usage telemetry aggregated: {prompt_tokens_total} prompt, "
                f"{completion_tokens_total} completion, {total_tokens} total tokens."
            )
        else:
            prompt_tokens_total = None
            completion_tokens_total = None
            total_tokens = None

        gen_latencies = [
            tc.model_response.latency_ms / 1000.0
            for tc in run.result.test_case_results
            if tc.model_response and tc.model_response.latency_ms is not None
        ]
        generation_duration_mean = (
            (sum(gen_latencies) / len(gen_latencies)) if gen_latencies else None
        )
        if generation_duration_mean is not None:
            notes.append(
                "Generation duration derived from adapter wall-clock latency measurements."
            )
    else:
        prompt_tokens_total = None
        completion_tokens_total = None
        total_tokens = None
        generation_duration_mean = None

    if not tokens_available:
        notes.append("Token usage telemetry is unavailable (adapter returned no token metrics).")
    notes.append("Monetary cost calculation is unavailable (no cost calculation or pricing table).")

    return ExecutionPerformanceSummary(
        run_id=run.run_id,
        wall_clock_duration_seconds=wall_clock,
        per_case_duration_seconds_mean=per_case_mean,
        retrieval_duration_seconds_mean=None,
        generation_duration_seconds_mean=generation_duration_mean,
        evaluator_duration_seconds_mean=evaluator_mean,
        tokens_available=tokens_available,
        cost_available=False,
        notes=notes,
        prompt_tokens_total=prompt_tokens_total,
        completion_tokens_total=completion_tokens_total,
        total_tokens=total_tokens,
    )
