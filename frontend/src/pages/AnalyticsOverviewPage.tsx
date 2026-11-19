import { Link } from "react-router-dom";
import { FixtureNotice } from "../components/FixtureNotice";
import { Card, SectionHeading, Stat } from "../components/Card";
import { Tag } from "../components/Tag";
import { formatTimestamp } from "../lib/derived";
import { toneForFailureCategory } from "../lib/tone";
import { useAnalyticsOverview, useExperiments, useTrends } from "../data/useData";
import { DataStateView } from "../components/DataStateView";
import type { FailureCategory } from "../types/domain";

export function AnalyticsOverviewPage() {
  const {
    data: overview,
    loading: ovLoading,
    error: ovError,
    refetch: refetchOv,
  } = useAnalyticsOverview();
  const {
    data: experiments,
    loading: expLoading,
    error: expError,
    refetch: refetchExp,
  } = useExperiments();

  const primaryExperiment = experiments?.[0];
  const {
    data: trend,
    loading: trendLoading,
    error: trendError,
    refetch: refetchTrend,
  } = useTrends(primaryExperiment?.experiment_id || "");

  const loading = ovLoading || expLoading || (Boolean(primaryExperiment) && trendLoading);
  const error = ovError || expError || trendError;

  const handleRetry = () => {
    void refetchOv();
    void refetchExp();
    void refetchTrend();
  };

  const failureCategories = Object.entries(
    overview?.primary_failure_distribution || {},
  ).filter(([, count]) => count > 0);

  return (
    <div>
      <SectionHeading
        eyebrow="Observability & Aggregates"
        title="Cross-Experiment Analytics"
        description="Consolidated analytical view of evaluation outcomes, failure distributions, retrieval quality, and longitudinal stability across controlled experimental conditions."
      />
      <FixtureNotice />

      <DataStateView loading={loading} error={error} onRetry={handleRetry}>
        {overview && (
          <>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Stat
                label="Total experiments"
                value={overview.total_experiments}
                subtext="Configured test definitions"
              />
              <Stat
                label="Total runs"
                value={overview.total_runs}
                subtext="Executed runs across experiments"
              />
              <Stat
                label="Test cases evaluated"
                value={overview.total_test_cases}
                subtext="Underlying sample size"
              />
              <Stat
                label="Evaluations recorded"
                value={overview.total_evaluations}
                subtext="Separate measurement procedures"
              />
              <Stat
                label="Evaluator methods"
                value={Object.keys(overview.evaluator_coverage).length}
                subtext="Distinct metric protocols"
              />
              <Stat
                label="Evaluator disagreements"
                value={overview.disagreement_count}
                subtext="Contradictory evaluator verdicts"
              />
              <Stat
                label="Primary failure modes"
                value={failureCategories.length}
                subtext="Distinct observed failure kinds"
              />
              <Stat
                label="Recent runs"
                value={overview.recent_runs.length}
                subtext="Latest executions indexed"
              />
            </div>

            <div className="mt-8 grid gap-6 lg:grid-cols-2">
              <Card>
                <div className="flex items-baseline justify-between">
                  <SectionHeading
                    eyebrow="Coverage"
                    title="Evaluator Coverage"
                    description="Total evaluation records per metric evaluator."
                  />
                </div>
                <div className="space-y-2">
                  {Object.entries(overview.evaluator_coverage).map(([name, count]) => {
                    const max = Math.max(...Object.values(overview.evaluator_coverage), 1);
                    const pct = Math.round((count / max) * 100);
                    return (
                      <div key={name} className="flex items-center gap-3 text-sm">
                        <span className="w-48 truncate font-mono text-xs">{name}</span>
                        <div className="h-2 flex-1 bg-paper-dim">
                          <div className="h-2 bg-ink" style={{ width: `${pct}%` }} />
                        </div>
                        <span className="w-12 text-right font-mono text-xs text-ink-soft">
                          {count}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </Card>

              <Card>
                <SectionHeading
                  eyebrow="Taxonomy"
                  title="Consolidated Failure Modes"
                  description="Primary failure distribution synthesized from all diagnosed runs."
                />
                {failureCategories.length === 0 ? (
                  <p className="text-xs text-ink-soft">No categorized failures in current runs.</p>
                ) : (
                  <div className="space-y-2">
                    {failureCategories.map(([category, count]) => {
                      const total = overview.total_test_cases || 1;
                      const pct = Number.isFinite(count) && total > 0 ? ((count / total) * 100).toFixed(1) : "0.0";
                      return (
                        <div key={category} className="flex items-center justify-between text-sm">
                          <div className="flex items-center gap-2">
                            <Tag tone={toneForFailureCategory(category as FailureCategory)}>
                              {category}
                            </Tag>
                          </div>
                          <div className="flex items-center gap-2 font-mono text-xs text-ink-soft">
                            <span>{count}</span>
                            <span>({pct}%)</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </Card>
            </div>

            {/* Longitudinal Trend Section */}
            {trend && trend.has_trend && (
              <div className="mt-8">
                <Card>
                  <div className="flex flex-col gap-2 sm:flex-row sm:items-baseline sm:justify-between">
                    <SectionHeading
                      eyebrow="Stability"
                      title={`Longitudinal Trajectory: ${primaryExperiment?.name ?? trend.experiment_id}`}
                      description={
                        trend.is_repeated_run_trend
                          ? "Trajectory over repeated executions under identical configuration. Variations reflect non-determinism."
                          : "Trajectory across runs with configuration transitions (marked below)."
                      }
                    />
                    <span className="font-mono text-xs text-ink-soft">
                      {trend.run_ids.length} consecutive runs
                    </span>
                  </div>

                  {trend.configuration_transitions.length > 0 && (
                    <div className="mb-4 border border-line bg-paper-dim px-3 py-2 text-xs">
                      <span className="font-mono uppercase tracking-wide">Notice:</span>{" "}
                      Configuration transition detected at run index{" "}
                      {trend.configuration_transitions.join(", ")}. Measurements before and after
                      reflect different experimental conditions and must not be interpreted as
                      spontaneous drift.
                    </div>
                  )}

                  <div className="space-y-4">
                    {Object.entries(trend.metric_series).map(([metric, points]) => (
                      <div key={metric} className="border-t border-line pt-3">
                        <div className="mb-2 flex items-center justify-between">
                          <span className="font-mono text-xs font-medium">{metric}</span>
                          <span className="text-xs text-ink-soft">
                            Min:{" "}
                            {Math.min(
                              ...(points.filter((p): p is number => p !== null) as number[]),
                            ).toFixed(3)}{" "}
                            | Max:{" "}
                            {Math.max(
                              ...(points.filter((p): p is number => p !== null) as number[]),
                            ).toFixed(3)}
                          </span>
                        </div>
                        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-6">
                          {points.map((val, idx) => (
                            <div
                              key={idx}
                              className={`border p-2 text-center ${
                                trend.configuration_transitions.includes(idx)
                                  ? "border-accent bg-paper-dim"
                                  : "border-line bg-paper"
                              }`}
                            >
                              <div className="font-mono text-[10px] text-ink-soft">
                                Run #{idx + 1}
                              </div>
                              <div className="font-mono text-xs font-semibold">
                                {val !== null ? val.toFixed(3) : "—"}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </Card>
              </div>
            )}

            <div className="mt-8">
              <Card>
                <SectionHeading
                  eyebrow="Provenance"
                  title="Indexed Experiment Runs"
                  description="Recent executions supplying data to this analytical layer."
                />
                <div className="overflow-x-auto border border-line">
                  <table className="w-full min-w-[700px] border-collapse text-sm">
                    <thead>
                      <tr className="border-b border-line bg-paper-dim text-left font-mono text-xs uppercase">
                        <th className="px-3 py-2">Run ID</th>
                        <th className="px-3 py-2">Experiment ID</th>
                        <th className="px-3 py-2">Cases</th>
                        <th className="px-3 py-2">Status</th>
                        <th className="px-3 py-2">Executed</th>
                        <th className="px-3 py-2 text-right">Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {overview.recent_runs.map((run) => (
                        <tr
                          key={run.run_id}
                          className="border-b border-line hover:bg-paper-dim/40 last:border-0"
                        >
                          <td className="px-3 py-2 font-mono text-xs font-medium">
                            <Link to={`/runs/${run.run_id}`} className="underline">
                              {run.run_id}
                            </Link>
                          </td>
                          <td className="px-3 py-2 text-xs">
                            <Link to={`/experiments/${run.experiment_id}`} className="underline">
                              {run.experiment_id}
                            </Link>
                          </td>
                          <td className="px-3 py-2 font-mono text-xs text-ink-soft">
                            {run.case_count}
                          </td>
                          <td className="px-3 py-2 font-mono text-xs text-ink-soft">
                            {run.status}
                          </td>
                          <td className="px-3 py-2 text-xs text-ink-soft">
                            {formatTimestamp(run.started_at)}
                          </td>
                          <td className="px-3 py-2 text-right">
                            <Link
                              to={`/runs/${run.run_id}`}
                              className="font-mono text-xs underline"
                            >
                              Inspect &rarr;
                            </Link>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            </div>
          </>
        )}
      </DataStateView>
    </div>
  );
}
