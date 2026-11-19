"""
Trend and longitudinal analysis across sequential experiment runs.

Tracks metric trajectories over chronological runs, detects configuration transitions,
and distinguishes repeated baseline trials from controlled experimental interventions.
"""

from __future__ import annotations

from typing import Any

from llm_reliability.analytics.aggregations import aggregate_evaluation, measure_performance
from llm_reliability.analytics.models import TrendAnalysis
from llm_reliability.experiments.models import ExperimentRun


def compute_trend_analysis(
    runs: list[ExperimentRun],
    experiment_id: str | None = None,
) -> TrendAnalysis:
    """Analyze longitudinal trends across chronologically ordered runs.

    Detects whether configuration changes occurred between runs, and marks
    ``has_trend=False`` if fewer than 2 runs are supplied.
    """
    if not runs:
        return TrendAnalysis(
            run_ids=[],
            experiment_id=experiment_id,
            is_repeated_run_trend=False,
            has_trend=False,
            configuration_transitions=[],
            metric_series={},
            sample_sizes=[],
            timestamps=[],
        )

    # Sort runs chronologically by start timestamp
    sorted_runs = sorted(runs, key=lambda r: r.started_at)
    run_ids = [r.run_id for r in sorted_runs]
    timestamps = [r.started_at for r in sorted_runs]
    sample_sizes = [len(r.result.test_case_results) if r.result else 0 for r in sorted_runs]

    has_trend = len(sorted_runs) >= 2

    # Check configuration transitions
    transitions: list[int] = []
    fingerprints = [r.config_fingerprint for r in sorted_runs]
    for i in range(1, len(fingerprints)):
        if fingerprints[i] != fingerprints[i - 1]:
            transitions.append(i)

    is_repeated_run_trend = len(set(fingerprints)) == 1

    # Extract metrics per run
    metric_series: dict[str, list[float | None]] = {}
    eval_aggregates: list[dict[str, Any]] = [aggregate_evaluation(r) for r in sorted_runs]
    perf_summaries = [measure_performance(r) for r in sorted_runs]

    all_evaluators: set[str] = set()
    for agg in eval_aggregates:
        all_evaluators.update(agg.keys())

    for eval_name in sorted(all_evaluators):
        pass_rate_key = f"{eval_name}_pass_rate"
        score_mean_key = f"{eval_name}_mean_score"
        metric_series[pass_rate_key] = []
        metric_series[score_mean_key] = []

        for agg in eval_aggregates:
            eval_data = agg.get(eval_name)
            if eval_data:
                metric_series[pass_rate_key].append(eval_data.pass_rate)
                metric_series[score_mean_key].append(eval_data.score_mean)
            else:
                metric_series[pass_rate_key].append(None)
                metric_series[score_mean_key].append(None)

    # Performance series
    metric_series["wall_clock_seconds"] = [p.wall_clock_duration_seconds for p in perf_summaries]
    metric_series["per_case_mean_seconds"] = [
        p.per_case_duration_seconds_mean for p in perf_summaries
    ]

    return TrendAnalysis(
        run_ids=run_ids,
        experiment_id=experiment_id,
        is_repeated_run_trend=is_repeated_run_trend,
        has_trend=has_trend,
        configuration_transitions=transitions,
        metric_series=metric_series,
        sample_sizes=sample_sizes,
        timestamps=timestamps,
    )
