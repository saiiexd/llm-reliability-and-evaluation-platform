"""
Experiment and run comparison mechanisms.

Enables rigorous cross-run comparisons of configurations, evaluation outcomes,
failure category distributions, retrieval metrics, robustness consistency, and
execution performance. Assesses methodological compatibility explicitly.
"""

from __future__ import annotations

from typing import Any

from llm_reliability.analytics.aggregations import (
    aggregate_evaluation,
    aggregate_failures,
    aggregate_retrieval,
    aggregate_robustness,
    measure_performance,
)
from llm_reliability.analytics.models import ComparisonCompatibility, RunComparison
from llm_reliability.diagnostics.diagnosis import ReliabilityDiagnostic
from llm_reliability.diagnostics.robustness import RobustnessResult
from llm_reliability.experiments.models import ExperimentRun
from llm_reliability.rag.evaluators import RetrievalEvaluationResult


def assess_compatibility(
    runs: list[ExperimentRun],
) -> tuple[ComparisonCompatibility, list[str], dict[str, Any]]:
    """Assess whether a set of runs is methodologically valid to compare directly.

    Returns the compatibility level, diagnostic reasons explaining the assessment,
    and a structured dictionary of configuration diffs across runs.
    """
    if not runs:
        return (
            ComparisonCompatibility.MATERIALLY_INCOMPATIBLE,
            ["No runs provided for comparison."],
            {},
        )

    if len(runs) == 1:
        return (
            ComparisonCompatibility.IDENTICAL_CONFIGURATION,
            ["Single run comparison baseline."],
            {"dataset_version": {runs[0].run_id: runs[0].dataset_version.version_id}},
        )

    reasons: list[str] = []

    # 1. Inspect dataset versions
    dataset_versions = {r.run_id: r.dataset_version.version_id for r in runs}
    distinct_datasets = set(dataset_versions.values())

    # 2. Inspect config fingerprints
    fingerprints = {r.run_id: r.config_fingerprint for r in runs}
    distinct_fingerprints = set(fingerprints.values())

    # 3. Inspect model configurations
    model_configs = {
        r.run_id: {
            "model_id": r.model_config.model_id,
            "temperature": r.model_config.temperature,
            "max_tokens": r.model_config.max_tokens,
            "extra_params": r.model_config.extra_params,
        }
        for r in runs
    }

    # 4. Inspect evaluator configs
    evaluator_sets = {
        r.run_id: sorted([e.evaluator_name for e in r.evaluator_configs]) for r in runs
    }

    # 5. Inspect execution adapters (what actually produced the answers). The
    # config fingerprint does not include the adapter, so two runs of the same
    # experiment on different adapters share a fingerprint but are different
    # systems, not repeated trials.
    adapters = {r.run_id: r.adapter_id for r in runs}
    known_adapters = {a for a in adapters.values() if a is not None}
    unknown_adapter_runs = sorted(rid for rid, a in adapters.items() if a is None)

    diffs: dict[str, Any] = {
        "dataset_versions": dataset_versions,
        "config_fingerprints": fingerprints,
        "model_configs": model_configs,
        "evaluators": evaluator_sets,
        "adapters": adapters,
    }

    # Methodological compatibility logic
    if len(distinct_datasets) > 1:
        reasons.append(
            f"Runs evaluate differing dataset versions ({', '.join(sorted(distinct_datasets))}). "
            "Direct aggregate score comparisons are methodologically invalid; outcomes must be "
            "analyzed separately."
        )
        return ComparisonCompatibility.MATERIALLY_INCOMPATIBLE, reasons, diffs

    if len(known_adapters) > 1:
        reasons.append(
            f"Runs share the dataset version but were executed by different adapters "
            f"({', '.join(sorted(known_adapters))}). Differences reflect the systems under "
            "test, not run-to-run variation."
        )
        return ComparisonCompatibility.PARTIALLY_DIFFERENT_CONFIGURATION, reasons, diffs

    if len(distinct_fingerprints) == 1:
        if unknown_adapter_runs and known_adapters:
            reasons.append(
                "The executing adapter was not recorded for "
                f"{', '.join(unknown_adapter_runs)} (older record); it cannot be confirmed "
                "that every run used the same system."
            )
        reasons.append(
            f"All {len(runs)} runs share identical configuration fingerprint "
            f"('{list(distinct_fingerprints)[0][:8]}...'). Variations represent stochasticity, "
            "system latency fluctuations, or repeated trial consistency."
        )
        return ComparisonCompatibility.IDENTICAL_CONFIGURATION, reasons, diffs

    # Shared dataset, differing configs -> controlled intervention
    reasons.append(
        f"All {len(runs)} runs evaluate identical dataset version "
        f"('{list(distinct_datasets)[0]}'), but vary in model parameters, prompt templates, "
        "or evaluator selections. Valid for controlled comparative analysis."
    )
    return ComparisonCompatibility.PARTIALLY_DIFFERENT_CONFIGURATION, reasons, diffs


def compare_runs(
    runs: list[ExperimentRun],
    diagnostics_by_run: dict[str, list[ReliabilityDiagnostic]] | None = None,
    retrieval_by_run: dict[str, list[RetrievalEvaluationResult]] | None = None,
    robustness_by_run: dict[str, list[RobustnessResult]] | None = None,
) -> RunComparison:
    """Perform side-by-side comparative analysis across two or more experiment runs.

    Aggregates evaluation results, failure categories, retrieval metrics, robustness
    consistency, and execution timings. Preserves underlying denominators and warns
    of configuration divergences.
    """
    source_run_ids = [r.run_id for r in runs]
    compatibility, reasons, config_diffs = assess_compatibility(runs)

    # 1. Evaluation Aggregates Comparison
    eval_aggregates_per_run: dict[str, dict[str, Any]] = {}
    all_evaluator_names: set[str] = set()

    for r in runs:
        agg = aggregate_evaluation(r)
        eval_aggregates_per_run[r.run_id] = {k: v.to_dict() for k, v in agg.items()}
        all_evaluator_names.update(agg.keys())

    eval_comparisons: dict[str, dict[str, Any]] = {}
    for eval_name in sorted(all_evaluator_names):
        pass_rates: dict[str, float | None] = {}
        numerators: dict[str, int] = {}
        denominators: dict[str, int] = {}
        score_means: dict[str, float | None] = {}

        for r in runs:
            r_data = eval_aggregates_per_run[r.run_id].get(eval_name)
            if r_data:
                pass_rates[r.run_id] = r_data["pass_rate"]
                numerators[r.run_id] = r_data["pass_rate_numerator"]
                denominators[r.run_id] = r_data["pass_rate_denominator"]
                score_means[r.run_id] = r_data["score_mean"]
            else:
                pass_rates[r.run_id] = None
                numerators[r.run_id] = 0
                denominators[r.run_id] = 0
                score_means[r.run_id] = None

        eval_comparisons[eval_name] = {
            "pass_rates": pass_rates,
            "numerators": numerators,
            "denominators": denominators,
            "score_means": score_means,
        }

    # 2. Failure Distribution Comparisons
    failure_comparisons: dict[str, dict[str, int]] = {}
    if diagnostics_by_run:
        for r_id in source_run_ids:
            diags = diagnostics_by_run.get(r_id, [])
            f_dist = aggregate_failures(diags, r_id)
            for cat_count in f_dist.primary_categories.values():
                cat_name = cat_count.category.value
                if cat_name not in failure_comparisons:
                    failure_comparisons[cat_name] = {}
                failure_comparisons[cat_name][r_id] = cat_count.count

    # 3. Retrieval Comparisons
    retrieval_comparisons: dict[str, dict[str, float | None]] = {}
    if retrieval_by_run:
        for r_id in source_run_ids:
            ret_results = retrieval_by_run.get(r_id, [])
            ret_sums = aggregate_retrieval(ret_results)
            for rs in ret_sums:
                metric_key = f"{rs.metric_name}_k{rs.k}" if rs.k is not None else rs.metric_name
                if metric_key not in retrieval_comparisons:
                    retrieval_comparisons[metric_key] = {}
                retrieval_comparisons[metric_key][r_id] = rs.mean_score

    # 4. Robustness Comparisons
    robustness_comparisons: dict[str, float | None] = {}
    if robustness_by_run:
        for r_id in source_run_ids:
            rob_results = robustness_by_run.get(r_id, [])
            if rob_results:
                r_sum = aggregate_robustness(rob_results)
                robustness_comparisons[r_id] = r_sum.consistency_rate
            else:
                robustness_comparisons[r_id] = None

    # 5. Performance Comparisons
    perf_comparisons: dict[str, Any] = {
        "wall_clock_seconds": {},
        "per_case_mean_seconds": {},
    }
    for r in runs:
        p_sum = measure_performance(r)
        perf_comparisons["wall_clock_seconds"][r.run_id] = p_sum.wall_clock_duration_seconds
        perf_comparisons["per_case_mean_seconds"][r.run_id] = p_sum.per_case_duration_seconds_mean

    return RunComparison(
        source_run_ids=source_run_ids,
        compatibility=compatibility,
        compatibility_reasons=reasons,
        configuration_diffs=config_diffs,
        evaluation_comparisons=eval_comparisons,
        failure_distribution_comparisons=failure_comparisons,
        retrieval_comparisons=retrieval_comparisons,
        robustness_comparisons=robustness_comparisons,
        performance_comparisons=perf_comparisons,
    )
