import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { FixtureNotice } from "../components/FixtureNotice";
import { Card, SectionHeading, Stat } from "../components/Card";
import { EmptyState } from "../components/EmptyState";
import { Tag } from "../components/Tag";
import { toneForFailureCategory, toneForRunStatus } from "../lib/tone";
import { formatTimestamp, shortFingerprint, testCaseSucceeded } from "../lib/derived";
import { useDiagnostics, useExperiment, useRun, useRunSummaries } from "../data/useData";
import { DataStateView } from "../components/DataStateView";
import type { FailureCategory } from "../types/domain";

export function RunDetailPage() {
  const { runId = "" } = useParams<{ runId: string }>();
  const [filterCategory, setFilterCategory] = useState<string>("all");

  const { data: run, loading: runLoading, error: runError, refetch: refetchRun } = useRun(runId);
  const { data: experiment } = useExperiment(run?.experiment_id || "");
  const { data: diagnosticsMap, loading: diagLoading } = useDiagnostics(runId);
  const { data: summary } = useRunSummaries(runId);

  const loading = runLoading || diagLoading;
  const diagnostics = diagnosticsMap || {};

  const testCases = run?.result?.test_case_results || [];
  const numTestCases = testCases.length;
  const numSucceeded = testCases.filter(testCaseSucceeded).length;
  const numFailed = numTestCases - numSucceeded;
  const disagreements = summary?.evaluation_summary?.disagreement_count ?? 0;

  // Category counts
  const categoryCounts = new Map<FailureCategory, number>();
  for (const tc of testCases) {
    const diag = diagnostics[tc.test_case_id];
    if (diag?.failure_category) {
      categoryCounts.set(
        diag.failure_category,
        (categoryCounts.get(diag.failure_category) ?? 0) + 1,
      );
    }
  }

  // Token & Telemetry Aggregations
  const tokensAvailable = testCases.some(
    (tc) =>
      tc.model_response &&
      (tc.model_response.prompt_tokens !== null || tc.model_response.completion_tokens !== null),
  );
  const promptTokensTotal = tokensAvailable
    ? testCases.reduce((acc, tc) => acc + (tc.model_response?.prompt_tokens ?? 0), 0)
    : null;
  const completionTokensTotal = tokensAvailable
    ? testCases.reduce((acc, tc) => acc + (tc.model_response?.completion_tokens ?? 0), 0)
    : null;
  const totalTokens = tokensAvailable
    ? testCases.reduce((acc, tc) => acc + (tc.model_response?.total_tokens ?? 0), 0)
    : null;

  const latencies = testCases
    .map((tc) => tc.model_response?.latency_ms)
    .filter((l): l is number => l !== null && l !== undefined);
  const meanLatencyMs = latencies.length > 0 ? latencies.reduce((a, b) => a + b, 0) / latencies.length : null;

  const filteredTestCases = testCases.filter((tc) => {
    if (filterCategory === "all") return true;
    const diag = diagnostics[tc.test_case_id];
    return diag?.failure_category === filterCategory;
  });

  return (
    <div>
      <DataStateView loading={loading} error={runError} onRetry={refetchRun}>
        {!run || !run.result ? (
          <div>
            <SectionHeading eyebrow="Run execution" title="Run not found" />
            <EmptyState
              title="Run not found"
              detail={`No experiment run with ID "${runId}" exists in the current repository.`}
            />
          </div>
        ) : (
          <>
            <SectionHeading
              eyebrow="Run execution"
              title={`Run ${shortFingerprint(run.run_id, 16)}`}
              action={
                experiment && (
                  <Link
                    to={`/experiments/${experiment.experiment_id}`}
                    className="font-mono text-xs text-ink underline"
                  >
                    View Experiment Definition &rarr;
                  </Link>
                )
              }
            />
            <FixtureNotice />

            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Stat label="Total test cases" value={numTestCases} />
              <Stat label="Execution succeeded" value={numSucceeded} />
              <Stat label="Execution failed" value={numFailed} />
              <Stat label="Evaluator disagreements" value={disagreements} />
              <Stat label="Started" value={formatTimestamp(run.started_at)} />
              <Stat
                label="Duration"
                value={
                  run.duration_seconds !== null && run.duration_seconds !== undefined
                    ? `${run.duration_seconds.toFixed(2)}s`
                    : "n/a"
                }
              />
              <Stat label="Dataset" value={run.dataset_version?.dataset_name || "N/A"} />
              <Stat label="Status" value={<Tag tone={toneForRunStatus(run.status)}>{run.status}</Tag>} />
            </div>

            <div className="mt-8 grid gap-6 lg:grid-cols-3">
              <Card>
                <SectionHeading eyebrow="Provenance" title="Run Metadata" />
                <dl className="space-y-2 text-sm">
                  <div className="flex items-baseline justify-between border-b border-line py-1">
                    <dt className="text-ink-soft">Run ID</dt>
                    <dd className="font-mono text-xs">{run.run_id}</dd>
                  </div>
                  <div className="flex items-baseline justify-between border-b border-line py-1">
                    <dt className="text-ink-soft">Experiment ID</dt>
                    <dd className="font-mono text-xs">{run.experiment_id}</dd>
                  </div>
                  <div className="flex items-baseline justify-between border-b border-line py-1">
                    <dt className="text-ink-soft">Config Fingerprint</dt>
                    <dd className="font-mono text-xs">{shortFingerprint(run.config_fingerprint)}</dd>
                  </div>
                  {run.benchmark_id && (
                    <div className="flex items-baseline justify-between border-b border-line py-1">
                      <dt className="text-ink-soft">Benchmark ID</dt>
                      <dd className="font-mono text-xs text-ink">
                        {run.benchmark_id} (rep #{run.repetition_index ?? 1})
                      </dd>
                    </div>
                  )}
                  <div className="flex items-baseline justify-between py-1">
                    <dt className="text-ink-soft">Evaluators Active</dt>
                    <dd className="font-mono text-xs">
                      {run.result.evaluator_names.join(", ")}
                    </dd>
                  </div>
                </dl>
              </Card>

              {/* Telemetry & Token Accounting Card */}
              <Card>
                <SectionHeading eyebrow="Telemetry" title="Tokens & Latency Provenance" />
                <dl className="space-y-2 text-sm">
                  <div className="flex items-baseline justify-between border-b border-line py-1">
                    <dt className="text-ink-soft">Token Accounting</dt>
                    <dd>
                      <Tag tone={tokensAvailable ? "neutral" : "muted"}>
                        {tokensAvailable ? "provider-reported" : "telemetry unavailable"}
                      </Tag>
                    </dd>
                  </div>
                  <div className="flex items-baseline justify-between border-b border-line py-1">
                    <dt className="text-ink-soft">Prompt Tokens</dt>
                    <dd className="font-mono text-xs">
                      {promptTokensTotal !== null ? promptTokensTotal.toLocaleString() : "—"}
                    </dd>
                  </div>
                  <div className="flex items-baseline justify-between border-b border-line py-1">
                    <dt className="text-ink-soft">Completion Tokens</dt>
                    <dd className="font-mono text-xs">
                      {completionTokensTotal !== null ? completionTokensTotal.toLocaleString() : "—"}
                    </dd>
                  </div>
                  <div className="flex items-baseline justify-between border-b border-line py-1">
                    <dt className="text-ink-soft">Total Tokens</dt>
                    <dd className="font-mono text-xs font-semibold">
                      {totalTokens !== null ? totalTokens.toLocaleString() : "—"}
                    </dd>
                  </div>
                  <div className="flex items-baseline justify-between py-1">
                    <dt className="text-ink-soft">Mean Model Latency</dt>
                    <dd className="font-mono text-xs">
                      {meanLatencyMs !== null ? `${meanLatencyMs.toFixed(1)}ms` : "—"}
                    </dd>
                  </div>
                </dl>
              </Card>

              <Card>
                <SectionHeading eyebrow="Distribution" title="Primary Failure Categories" />
                {categoryCounts.size === 0 ? (
                  <EmptyState title="No failures categorized for this run." />
                ) : (
                  <ul className="space-y-2">
                    {[...categoryCounts.entries()].map(([cat, count]) => (
                      <li key={cat} className="flex items-center justify-between text-sm">
                        <Tag tone={toneForFailureCategory(cat)}>{cat}</Tag>
                        <span className="font-mono text-xs text-ink-soft">{count} cases</span>
                      </li>
                    ))}
                  </ul>
                )}
              </Card>
            </div>

            <div className="mt-8">
              <Card>
                <div className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                  <SectionHeading eyebrow="Case records" title="Evaluated Test Cases" />
                  <div className="flex items-center gap-2">
                    <label htmlFor="cat-filter" className="text-xs text-ink-soft">
                      Filter category:
                    </label>
                    <select
                      id="cat-filter"
                      value={filterCategory}
                      onChange={(e) => setFilterCategory(e.target.value)}
                      className="border border-line bg-paper px-2 py-1 text-xs text-ink focus:border-ink focus:outline-none"
                    >
                      <option value="all">All ({numTestCases})</option>
                      {[...categoryCounts.entries()].map(([cat, count]) => (
                        <option key={cat} value={cat}>
                          {cat} ({count})
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                <div className="overflow-x-auto border border-line">
                  <table className="w-full min-w-[700px] border-collapse text-sm">
                    <thead>
                      <tr className="border-b border-line bg-paper-dim text-left font-mono text-xs uppercase">
                        <th className="px-3 py-2">Test Case ID</th>
                        <th className="px-3 py-2">Execution</th>
                        <th className="px-3 py-2">Primary Category</th>
                        <th className="px-3 py-2">Evaluator Disagreement</th>
                        <th className="px-3 py-2">Tokens</th>
                        <th className="px-3 py-2">Latency</th>
                        <th className="px-3 py-2">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredTestCases.map((tc) => {
                        const diag = diagnostics[tc.test_case_id];
                        const passed = testCaseSucceeded(tc);
                        const hasDisagreement =
                          diag?.secondary_findings?.some((f) =>
                            f.toLowerCase().includes("evaluator disagreement"),
                          ) ?? false;
                        const respTokens = tc.model_response?.total_tokens;
                        const respLat = tc.model_response?.latency_ms;

                        return (
                          <tr
                            key={tc.test_case_id}
                            className="border-b border-line last:border-0 hover:bg-paper-dim"
                          >
                            <td className="px-3 py-2 font-mono text-xs">{tc.test_case_id}</td>
                            <td className="px-3 py-2">
                              <Tag tone={passed ? "neutral" : "negative"}>
                                {passed ? "succeeded" : "failed"}
                              </Tag>
                            </td>
                            <td className="px-3 py-2">
                              {diag?.failure_category ? (
                                <Tag tone={toneForFailureCategory(diag.failure_category)}>
                                  {diag.failure_category}
                                </Tag>
                              ) : (
                                <Tag tone="muted">undetermined</Tag>
                              )}
                            </td>
                            <td className="px-3 py-2">
                              {hasDisagreement ? (
                                <Tag tone="negative">disagreement</Tag>
                              ) : (
                                <Tag tone="muted">unanimous</Tag>
                              )}
                            </td>
                            <td className="px-3 py-2 font-mono text-xs text-ink-soft">
                              {respTokens !== undefined && respTokens !== null ? `${respTokens} tok` : "—"}
                            </td>
                            <td className="px-3 py-2 font-mono text-xs text-ink-soft">
                              {respLat !== undefined && respLat !== null ? `${respLat.toFixed(0)}ms` : "—"}
                            </td>
                            <td className="px-3 py-2">
                              <Link
                                to={`/runs/${run.run_id}/cases/${tc.test_case_id}`}
                                className="font-mono text-xs text-ink underline hover:text-ink-soft"
                              >
                                View Forensic Analysis &rarr;
                              </Link>
                            </td>
                          </tr>
                        );
                      })}
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
