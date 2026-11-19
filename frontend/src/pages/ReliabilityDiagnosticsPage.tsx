import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { FixtureNotice } from "../components/FixtureNotice";
import { Card, SectionHeading, Stat } from "../components/Card";
import { EmptyState } from "../components/EmptyState";
import { Tag } from "../components/Tag";
import { toneForDiagnosticStatus, toneForFailureCategory } from "../lib/tone";
import { shortFingerprint } from "../lib/derived";
import { useConnection } from "../context/ConnectionContext";
import { useRuns } from "../data/useData";
import * as fixtures from "../data/dataAccess";
import { DataStateView } from "../components/DataStateView";
import { FailureDistributionChart } from "../components/charts/FailureDistributionChart";
import type { FailureCategory, ReliabilityDiagnostic } from "../types/domain";

export function ReliabilityDiagnosticsPage() {
  const [selectedCategory, setSelectedCategory] = useState<string>("all");
  const { dataMode, client } = useConnection();
  const { data: runs, loading: runsLoading, error: runsError, refetch } = useRuns();

  interface DiagnosticRow extends ReliabilityDiagnostic {
    runId: string;
  }

  const [allDiagnostics, setAllDiagnostics] = useState<DiagnosticRow[]>([]);
  const [loadingDetails, setLoadingDetails] = useState<boolean>(false);
  const [detailError, setDetailError] = useState<Error | null>(null);

  useEffect(() => {
    async function loadData() {
      if (!runs) return;

      if (dataMode === "fixture") {
        const rows: DiagnosticRow[] = [];
        for (const run of runs) {
          const diagMap = fixtures.getDiagnosticsForRun(run.run_id);
          for (const [tcId, diag] of Object.entries(diagMap)) {
            rows.push({
              ...diag,
              runId: run.run_id,
              test_case_id: tcId,
            });
          }
        }
        setAllDiagnostics(rows);
        return;
      }

      // Live mode
      setLoadingDetails(true);
      setDetailError(null);
      try {
        const rows: DiagnosticRow[] = [];
        for (const run of runs) {
          try {
            const res = await client.listDiagnostics({ runId: run.run_id, limit: 100 });
            for (const item of res.items) {
              rows.push({
                ...item,
                runId: run.run_id,
              });
            }
          } catch {
            // Optional
          }
        }
        setAllDiagnostics(rows);
      } catch (err: unknown) {
        setDetailError(err instanceof Error ? err : new Error(String(err)));
      } finally {
        setLoadingDetails(false);
      }
    }

    void loadData();
  }, [runs, dataMode, client]);

  const categoryCounts = new Map<FailureCategory, number>();
  let determinedCount = 0;
  let undeterminedCount = 0;
  let secondaryFindingsCount = 0;

  for (const diag of allDiagnostics) {
    if (diag.status === "determined") determinedCount++;
    if (diag.status === "undetermined") undeterminedCount++;
    if (diag.secondary_findings?.length) secondaryFindingsCount += diag.secondary_findings.length;

    if (diag.failure_category) {
      categoryCounts.set(
        diag.failure_category,
        (categoryCounts.get(diag.failure_category) ?? 0) + 1,
      );
    }
  }

  const filteredDiagnostics = allDiagnostics.filter((d) => {
    if (selectedCategory === "all") return true;
    return d.failure_category === selectedCategory;
  });

  const loading = runsLoading || loadingDetails;
  const error = runsError || detailError;

  return (
    <div>
      <SectionHeading eyebrow="Consolidated failure attribution" title="Reliability Diagnostics" />
      <FixtureNotice />

      <DataStateView loading={loading} error={error} onRetry={refetch}>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat label="Total Diagnoses" value={allDiagnostics.length} />
          <Stat label="Determined Outcomes" value={determinedCount} />
          <Stat label="Undetermined / Gaps" value={undeterminedCount} />
          <Stat label="Preserved Secondary Findings" value={secondaryFindingsCount} />
        </div>

        <div className="mt-8">
          <Card>
            <SectionHeading eyebrow="Distribution" title="Primary Failure Categories" />
            <p className="mb-4 text-xs text-ink-soft">
              Failure categories synthesized from correctness, faithfulness, and retrieval evidence
              following strict scientific precedence. Secondary findings are preserved alongside
              the primary category.
            </p>

            {categoryCounts.size === 0 ? (
              <EmptyState title="No reliability diagnoses recorded." />
            ) : (
              <div className="space-y-6">
                <FailureDistributionChart
                  categoryCounts={categoryCounts}
                  totalFailures={allDiagnostics.filter((d) => d.failure_category && d.failure_category.toLowerCase() !== "correct").length}
                  title="Failure Mode Breakdown"
                />

                <div>
                  <p className="text-xs font-mono uppercase tracking-wider text-ink-soft mb-2">Category Filter Cards</p>
                  <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                    {[...categoryCounts.entries()].map(([cat, count]) => (
                      <button
                        key={cat}
                        onClick={() => setSelectedCategory(selectedCategory === cat ? "all" : cat)}
                        className={`border p-3 text-left transition-colors ${
                          selectedCategory === cat
                            ? "border-ink bg-paper-dim ring-1 ring-ink"
                            : "border-line bg-paper hover:bg-paper-dim/40"
                        }`}
                      >
                        <div className="mb-1">
                          <Tag tone={toneForFailureCategory(cat)}>{cat}</Tag>
                        </div>
                        <p className="font-display text-xl">{count}</p>
                        <p className="text-[11px] text-ink-soft">
                          {selectedCategory === cat ? "Filtered (click to clear)" : "Click to filter"}
                        </p>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </Card>
        </div>

        <div className="mt-8">
          <Card>
            <div className="mb-4 flex flex-col gap-2 border-b border-line pb-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <p className="font-mono text-xs uppercase tracking-wide text-ink-soft">Inspection</p>
                <h2 className="font-display text-xl">Diagnostic Cases</h2>
              </div>
              <div className="flex items-center gap-2">
                <label htmlFor="cat-filter" className="text-xs text-ink-soft">
                  Filter:
                </label>
                <select
                  id="cat-filter"
                  value={selectedCategory}
                  onChange={(e) => setSelectedCategory(e.target.value)}
                  className="border border-line bg-paper px-2 py-1 font-mono text-xs text-ink"
                >
                  <option value="all">All Categories ({allDiagnostics.length})</option>
                  {[...categoryCounts.keys()].map((cat) => (
                    <option key={cat} value={cat}>
                      {cat} ({categoryCounts.get(cat)})
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {filteredDiagnostics.length === 0 ? (
              <EmptyState title="No diagnostic records match the selected filter." />
            ) : (
              <div className="overflow-x-auto border border-line">
                <table className="w-full min-w-[760px] border-collapse text-sm">
                  <thead>
                    <tr className="border-b border-line bg-paper-dim text-left font-mono text-xs uppercase">
                      <th className="px-3 py-2">Test Case</th>
                      <th className="px-3 py-2">Run ID</th>
                      <th className="px-3 py-2">Primary Diagnosis</th>
                      <th className="px-3 py-2">Status</th>
                      <th className="px-3 py-2">Secondary Findings</th>
                      <th className="px-3 py-2">Confidence</th>
                      <th className="px-3 py-2 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredDiagnostics.map((d, idx) => (
                      <tr
                        key={idx}
                        className="border-b border-line hover:bg-paper-dim/40 last:border-0"
                      >
                        <td className="px-3 py-2 font-mono text-xs font-medium">
                          <Link
                            to={`/runs/${d.runId}/test-cases/${d.test_case_id}`}
                            className="underline"
                          >
                            {d.test_case_id}
                          </Link>
                        </td>
                        <td className="px-3 py-2 font-mono text-xs text-ink-soft">
                          <Link to={`/runs/${d.runId}`} className="underline">
                            {shortFingerprint(d.runId, 12)}
                          </Link>
                        </td>
                        <td className="px-3 py-2">
                          {d.failure_category ? (
                            <Tag tone={toneForFailureCategory(d.failure_category)}>
                              {d.failure_category}
                            </Tag>
                          ) : (
                            <span className="font-mono text-xs text-ink-soft">none</span>
                          )}
                        </td>
                        <td className="px-3 py-2">
                          <Tag tone={toneForDiagnosticStatus(d.status)}>{d.status}</Tag>
                        </td>
                        <td className="px-3 py-2">
                          {d.secondary_findings?.length ? (
                            <Tag tone="caution">{`${d.secondary_findings.length} findings`}</Tag>
                          ) : (
                            <span className="text-xs text-ink-soft">none</span>
                          )}
                        </td>
                        <td className="px-3 py-2 font-mono text-xs text-ink-soft">
                          {d.confidence !== null && d.confidence !== undefined
                            ? `${(d.confidence * 100).toFixed(0)}%`
                            : "n/a"}
                        </td>
                        <td className="px-3 py-2 text-right">
                          <Link
                            to={`/runs/${d.runId}/test-cases/${d.test_case_id}`}
                            className="font-mono text-xs underline"
                          >
                            Investigate &rarr;
                          </Link>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </div>
      </DataStateView>
    </div>
  );
}
