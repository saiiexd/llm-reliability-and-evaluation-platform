import { useState } from "react";
import { Link } from "react-router-dom";
import { FixtureNotice } from "../components/FixtureNotice";
import { Card, SectionHeading } from "../components/Card";
import { Tag } from "../components/Tag";
import { toneForFailureCategory, toneForRunStatus } from "../lib/tone";
import { formatCategoryLabel, formatTimestamp, testCasePassed } from "../lib/derived";
import { useAnalyticsOverview, useExperiments, useRuns } from "../data/useData";
import { DataStateView } from "../components/DataStateView";
import { PassFailDonutChart } from "../components/charts/PassFailDonutChart";
import { VisualMetricsBar } from "../components/charts/VisualMetricsBar";
import type { FailureCategory } from "../types/domain";

export function OverviewPage() {
  const {
    data: experiments,
    loading: expLoading,
    error: expError,
    refetch: refetchExp,
  } = useExperiments();
  const {
    data: runs,
    loading: runsLoading,
    error: runsError,
    refetch: refetchRuns,
  } = useRuns();
  const {
    data: overview,
    loading: ovLoading,
    error: ovError,
    refetch: refetchOv,
  } = useAnalyticsOverview();

  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);

  const loading = expLoading || runsLoading || ovLoading;
  const error = expError || runsError || ovError;

  const handleRetry = () => {
    void refetchExp();
    void refetchRuns();
    void refetchOv();
  };

  const expList = experiments || [];
  const runList = runs || [];

  // Identify most recent experiment and its latest run
  const mostRecentExp = expList.length > 0 ? expList[0] : null;
  const runsForRecentExp = mostRecentExp
    ? runList.filter((r) => r.experiment_id === mostRecentExp.experiment_id)
    : [];
  const latestRun = runsForRecentExp.length > 0 ? runsForRecentExp[0] : (runList[0] ?? null);

  // Latest run pass counts
  const latestTestCases = latestRun?.result?.test_case_results ?? [];
  const latestPassed = latestTestCases.filter(testCasePassed).length;
  const latestFailed = latestTestCases.length - latestPassed;

  // High-level summary metrics
  const totalTestCases =
    overview?.total_test_cases ??
    runList.reduce((acc, r) => acc + (r.result?.test_case_results.length ?? 0), 0);
  const totalPassed = runList.reduce(
    (acc, r) => acc + (r.result?.test_case_results.filter(testCasePassed).length ?? 0),
    0,
  );
  const passRateMean =
    totalTestCases > 0 ? `${((totalPassed / totalTestCases) * 100).toFixed(1)}%` : null;

  // Breakdown metrics for expandable section
  const evaluatorCoverage = overview?.evaluator_coverage
    ? Object.entries(overview.evaluator_coverage)
    : [];
  const failureDistribution = overview?.primary_failure_distribution
    ? Object.entries(overview.primary_failure_distribution)
    : [];

  return (
    <div className="space-y-6">
      {/* Streamlined Academic Introduction & Primary Action */}
      <section className="border-b border-line pb-6">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="max-w-2xl">
            <span className="font-mono text-xs uppercase tracking-wider text-ink-soft block mb-1">
              LLM Reliability & Evaluation Platform
            </span>
            <h1 className="font-serif text-2xl font-normal text-ink sm:text-3xl leading-snug">
              A scientific research environment for measuring, comparing, and diagnosing language model reliability.
            </h1>
            <p className="mt-2 text-xs text-ink-soft leading-relaxed">
              Measure answer correctness against references, trace context retrieval in RAG pipelines, and isolate failure causes under controlled experimental conditions.
            </p>
          </div>

          {/* Obvious Primary Action Buttons */}
          <div className="flex items-center gap-3 shrink-0">
            <Link
              to="/experiments?new=true"
              className="bg-ink px-5 py-2.5 font-mono text-xs font-semibold text-paper hover:bg-ink-soft transition-colors shadow-xs"
              aria-label="Start an Experiment"
            >
              Start an Experiment &rarr;
            </Link>
            <Link
              to="/results"
              className="border border-line bg-paper px-4 py-2.5 font-mono text-xs text-ink hover:bg-paper-dim transition-colors"
            >
              Explore Results &rarr;
            </Link>
          </div>
        </div>

        {/* Academic Research Lifecycle Bar */}
        <div className="mt-6 pt-4 border-t border-line/60">
          <p className="text-[10px] font-mono uppercase tracking-wider text-ink-soft mb-2">Standard Research Lifecycle</p>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
            <div className="border border-line/60 bg-paper-dim/40 p-2">
              <span className="font-mono text-[10px] text-ink-soft block">01 / OVERVIEW</span>
              <span className="text-ink font-medium">Platform Health & KPIs</span>
            </div>
            <div className="border border-line/60 bg-paper-dim/40 p-2">
              <span className="font-mono text-[10px] text-ink-soft block">02 / EXPERIMENTS</span>
              <span className="text-ink font-medium">Dataset & Model Definition</span>
            </div>
            <div className="border border-line/60 bg-paper-dim/40 p-2">
              <span className="font-mono text-[10px] text-ink-soft block">03 / RESULTS</span>
              <span className="text-ink font-medium">Pass Rates & Failure Modes</span>
            </div>
            <div className="border border-line/60 bg-paper-dim/40 p-2">
              <span className="font-mono text-[10px] text-ink-soft block">04 / ANALYSIS</span>
              <span className="text-ink font-medium">Comparison & Diagnostics</span>
            </div>
          </div>
        </div>
      </section>

      <FixtureNotice />

      <DataStateView loading={loading} error={error} onRetry={handleRetry}>
        {/* Simple Visual Summary of Project State (4 KPI Cards) */}
        <section>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Card className="flex flex-col justify-between p-4 bg-paper border border-line">
              <span className="font-mono text-[11px] uppercase tracking-wide text-ink-soft">
                Defined Experiments
              </span>
              <span className="font-serif text-3xl font-normal text-ink mt-2">
                {expList.length}
              </span>
              <span className="text-[11px] text-ink-soft mt-1">Configured benchmarks</span>
            </Card>

            <Card className="flex flex-col justify-between p-4 bg-paper border border-line">
              <span className="font-mono text-[11px] uppercase tracking-wide text-ink-soft">
                Executed Runs
              </span>
              <span className="font-serif text-3xl font-normal text-ink mt-2">
                {runList.length}
              </span>
              <span className="text-[11px] text-ink-soft mt-1">Completed executions</span>
            </Card>

            <Card className="flex flex-col justify-between p-4 bg-paper border border-line">
              <span className="font-mono text-[11px] uppercase tracking-wide text-ink-soft">
                Evaluated Test Cases
              </span>
              <span className="font-serif text-3xl font-normal text-ink mt-2">
                {totalTestCases}
              </span>
              <span className="text-[11px] text-ink-soft mt-1">Total model prompts scored</span>
            </Card>

            <Card className="flex flex-col justify-between p-4 bg-paper border border-line">
              <span className="font-mono text-[11px] uppercase tracking-wide text-ink-soft">
                Overall Pass Rate
              </span>
              <span className="font-serif text-3xl font-semibold text-ink mt-2">
                {passRateMean ?? "N/A"}
              </span>
              <span className="text-[11px] text-ink-soft mt-1">
                {totalPassed} of {totalTestCases} passed evaluators
              </span>
            </Card>
          </div>
        </section>

        {/* Latest Experiment Spotlight with Compact Visual Chart */}
        {mostRecentExp && (
          <section className="border border-line bg-paper p-5">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between border-b border-line pb-4 mb-4">
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-mono text-[11px] uppercase tracking-wider text-ink-soft">
                    Latest Experiment Spotlight
                  </span>
                  {latestRun && (
                    <Tag tone={toneForRunStatus(latestRun.status)}>{latestRun.status}</Tag>
                  )}
                </div>
                <h3 className="font-serif text-xl font-medium text-ink mt-1">
                  {mostRecentExp.name}
                </h3>
                <p className="font-mono text-xs text-ink-soft mt-0.5">
                  ID: {mostRecentExp.experiment_id} &bull; Created: {formatTimestamp(mostRecentExp.created_at)}
                </p>
              </div>

              <div className="flex items-center gap-3">
                <Link
                  to={`/experiments/${mostRecentExp.experiment_id}`}
                  className="font-mono text-xs text-ink underline hover:text-ink-soft"
                >
                  View Definition &rarr;
                </Link>
                {latestRun && (
                  <Link
                    to={`/results/${latestRun.run_id}`}
                    className="bg-paper-dim border border-line px-3.5 py-1.5 font-mono text-xs font-medium text-ink hover:bg-paper transition-colors"
                  >
                    Inspect Results &rarr;
                  </Link>
                )}
              </div>
            </div>

            {/* Visual Grid: Config Details on Left, Compact Donut Chart on Right */}
            <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-center">
              <div className="md:col-span-6 space-y-3 text-xs">
                <div className="grid grid-cols-2 gap-3">
                  <div className="border border-line p-3 bg-paper-dim">
                    <span className="block font-mono text-[10px] uppercase text-ink-soft">Target Model</span>
                    <span className="font-mono font-semibold text-ink text-sm">
                      {mostRecentExp.model_config?.model_id || "Unspecified"}
                    </span>
                  </div>
                  <div className="border border-line p-3 bg-paper-dim">
                    <span className="block font-mono text-[10px] uppercase text-ink-soft">Dataset</span>
                    <span className="font-medium text-ink text-sm">
                      {mostRecentExp.dataset_version?.dataset_name || "N/A"}
                    </span>
                    <span className="text-[11px] text-ink-soft block mt-0.5">
                      {mostRecentExp.dataset_version?.test_case_count ??
                        mostRecentExp.dataset_version?.test_cases?.length ??
                        mostRecentExp.test_case_count ??
                        0}{" "}
                      test cases
                    </span>
                  </div>
                </div>

                {latestRun && (
                  <div className="pt-2">
                    <span className="block font-mono text-[11px] uppercase text-ink-soft mb-1.5">
                      Latest Run Pass / Fail Distribution:
                    </span>
                    <VisualMetricsBar
                      passed={latestPassed}
                      failed={latestFailed}
                      total={latestTestCases.length}
                    />
                  </div>
                )}
              </div>

              {/* Compact Donut Visual */}
              <div className="md:col-span-6 flex justify-center border-t md:border-t-0 md:border-l border-line pt-4 md:pt-0 md:pl-6">
                {latestRun && latestTestCases.length > 0 ? (
                  <PassFailDonutChart
                    passed={latestPassed}
                    failed={latestFailed}
                    total={latestTestCases.length}
                    size={150}
                    thickness={18}
                    subtitle="Latest run outcome distribution"
                  />
                ) : (
                  <div className="text-center py-6 text-xs text-ink-soft">
                    No run executed yet for this experiment.
                  </div>
                )}
              </div>
            </div>
          </section>
        )}

        {/* Recent Experiments Clean Table */}
        <section>
          <div className="flex items-center justify-between mb-3">
            <div>
              <span className="font-mono text-[11px] uppercase tracking-wider text-ink-soft">
                Benchmark Registry
              </span>
              <h3 className="font-serif text-lg font-medium text-ink">Recent Experiments</h3>
            </div>
            <Link
              to="/experiments"
              className="font-mono text-xs text-ink underline hover:text-ink-soft"
            >
              View All ({expList.length}) &rarr;
            </Link>
          </div>

          <div className="border border-line bg-paper overflow-x-auto">
            {expList.length === 0 ? (
              <div className="p-6 text-center text-xs text-ink-soft">
                No experiments defined yet. Click &ldquo;+ Create Experiment&rdquo; to begin.
              </div>
            ) : (
              <table className="w-full text-left text-xs font-sans">
                <thead>
                  <tr className="border-b border-line bg-paper-dim uppercase font-mono text-[11px] text-ink-soft">
                    <th className="p-3">Experiment Name</th>
                    <th className="p-3">Dataset</th>
                    <th className="p-3">Model</th>
                    <th className="p-3">Evaluators</th>
                    <th className="p-3">Created</th>
                    <th className="p-3 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line">
                  {expList.slice(0, 5).map((exp) => {
                    const expRuns = runList.filter((r) => r.experiment_id === exp.experiment_id);
                    const latest = expRuns[0];

                    return (
                      <tr key={exp.experiment_id} className="hover:bg-paper-dim transition-colors">
                        <td className="p-3">
                          <Link
                            to={`/experiments/${exp.experiment_id}`}
                            className="font-medium text-ink hover:underline"
                          >
                            {exp.name}
                          </Link>
                          <span className="block font-mono text-[10px] text-ink-soft">
                            {exp.experiment_id}
                          </span>
                        </td>
                        <td className="p-3 font-mono text-[11px] text-ink">
                          {exp.dataset_version?.dataset_name}
                          <span className="block text-[10px] text-ink-soft">
                            {exp.dataset_version?.test_case_count ??
                              exp.dataset_version?.test_cases?.length ??
                              exp.test_case_count ??
                              0}{" "}
                            cases
                          </span>
                        </td>
                        <td className="p-3 font-mono text-[11px] text-ink">
                          {exp.model_config?.model_id || "Unspecified"}
                        </td>
                        <td className="p-3 font-mono text-[11px] text-ink-soft">
                          {exp.evaluator_configs?.map((e) => e.evaluator_name).join(", ") || "none"}
                        </td>
                        <td className="p-3 text-ink-soft text-[11px]">
                          {formatTimestamp(exp.created_at)}
                        </td>
                        <td className="p-3 text-right font-mono text-[11px]">
                          {latest ? (
                            <Link
                              to={`/results/${latest.run_id}`}
                              className="text-ink underline hover:text-ink-soft mr-3"
                            >
                              Results
                            </Link>
                          ) : null}
                          <Link
                            to={`/experiments/${exp.experiment_id}`}
                            className="text-ink-soft hover:text-ink"
                          >
                            Details &rarr;
                          </Link>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
        </section>

        {/* Expandable Technical Research Telemetry (Collapsed by Default) */}
        <section className="border-t border-line pt-4">
          <div className="flex items-center justify-between">
            <div>
              <h4 className="font-serif text-sm font-medium text-ink">Detailed Research Telemetry</h4>
              <p className="text-xs text-ink-soft">
                Evaluator coverage distributions, primary failure taxonomy, and longitudinal system history.
              </p>
            </div>
            <button
              type="button"
              onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
              className="border border-line bg-paper px-3 py-1 font-mono text-xs text-ink hover:bg-paper-dim transition-colors"
            >
              {showTechnicalDetails ? "Hide Technical Details ▲" : "View Technical Details ▼"}
            </button>
          </div>

          {showTechnicalDetails && (
            <div className="mt-4 grid gap-4 sm:grid-cols-2 text-xs">
              <Card>
                <SectionHeading eyebrow="Telemetry" title="Evaluator Coverage" />
                {evaluatorCoverage.length === 0 ? (
                  <p className="text-ink-soft">No evaluator data recorded yet.</p>
                ) : (
                  <ul className="divide-y divide-line font-mono text-xs mt-2">
                    {evaluatorCoverage.map(([name, count]) => (
                      <li key={name} className="flex justify-between py-1.5">
                        <span className="text-ink">{name}</span>
                        <span className="text-ink-soft">{count} evaluated</span>
                      </li>
                    ))}
                  </ul>
                )}
              </Card>

              <Card>
                <SectionHeading eyebrow="Telemetry" title="Observed Failure Modes" />
                {failureDistribution.length === 0 ? (
                  <p className="text-ink-soft">No failures classified yet.</p>
                ) : (
                  <ul className="divide-y divide-line font-mono text-xs mt-2">
                    {failureDistribution.map(([cat, count]) => (
                      <li key={cat} className="flex items-center justify-between py-1.5">
                        <span className="text-ink">{formatCategoryLabel(cat)}</span>
                        <Tag tone={toneForFailureCategory(cat as FailureCategory)}>{count}</Tag>
                      </li>
                    ))}
                  </ul>
                )}
              </Card>
            </div>
          )}
        </section>
      </DataStateView>
    </div>
  );
}
