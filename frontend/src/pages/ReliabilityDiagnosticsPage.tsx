import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { EmptyState } from "../components/EmptyState";
import { CATEGORY_HELP, formatCategoryLabel, formatTimestamp } from "../lib/derived";
import { useConnection } from "../context/ConnectionContext";
import { useRuns } from "../data/useData";
import * as fixtures from "../data/dataAccess";
import { DataStateView } from "../components/DataStateView";
import type { FailureCategory, ReliabilityDiagnostic } from "../types/domain";

export function ReliabilityDiagnosticsPage() {
  // "failing" = every diagnosis except "correct"; otherwise a single category.
  const [selectedCategory, setSelectedCategory] = useState<string>("failing");
  const [visibleCount, setVisibleCount] = useState(15);
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
  for (const diag of allDiagnostics) {
    if (diag.failure_category) {
      categoryCounts.set(diag.failure_category, (categoryCounts.get(diag.failure_category) ?? 0) + 1);
    }
  }
  const correctCount = categoryCounts.get("correct") ?? 0;
  const causes = [...categoryCounts.entries()].filter(([c]) => c !== "correct").sort((a, b) => b[1] - a[1]);
  const failingTotal = causes.reduce((n, [, c]) => n + c, 0);
  const maxCause = causes[0]?.[1] ?? 1;

  const filteredDiagnostics = allDiagnostics.filter((d) =>
    selectedCategory === "failing"
      ? d.failure_category && d.failure_category !== "correct"
      : d.failure_category === selectedCategory,
  );
  const runStarted = new Map((runs ?? []).map((r) => [r.run_id, r.started_at]));

  const loading = runsLoading || loadingDetails;
  const error = runsError || detailError;

  return (
    <DataStateView loading={loading} error={error} onRetry={refetch}>
      {allDiagnostics.length === 0 ? (
        <EmptyState title="No diagnosed answers yet." detail="Run an experiment to get diagnoses." />
      ) : (
        <div className="space-y-6">
          <section className="rounded-xl border border-line p-6">
            <h3 className="font-semibold">Causes</h3>
            <p className="mt-0.5 text-sm text-ink-soft">
              {failingTotal} of {allDiagnostics.length} diagnosed answers had a problem. {correctCount} were judged
              correct. Click a cause to list its answers.
            </p>
            <ul className="mt-4 space-y-1">
              {causes.map(([cat, count]) => {
                const active = selectedCategory === cat;
                return (
                  <li key={cat}>
                    <button
                      type="button"
                      onClick={() => {
                        setSelectedCategory(active ? "failing" : cat);
                        setVisibleCount(15);
                      }}
                      aria-pressed={active}
                      className={`w-full rounded-lg px-3 py-2.5 text-left transition-colors ${
                        active ? "bg-paper-raised" : "hover:bg-paper-dim"
                      }`}
                    >
                      <div className="flex items-baseline justify-between gap-4 text-sm">
                        <span className="font-medium">{formatCategoryLabel(cat)}</span>
                        <span className="tabular-nums text-ink-soft">{count}</span>
                      </div>
                      <p className="text-xs text-ink-soft">{CATEGORY_HELP[cat] ?? ""}</p>
                      <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-paper-raised">
                        <div className="h-full bg-accent" style={{ width: `${(count / maxCause) * 100}%` }} />
                      </div>
                    </button>
                  </li>
                );
              })}
            </ul>
          </section>

          <section className="rounded-xl border border-line p-6">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <h3 className="font-semibold">
                {selectedCategory === "failing" ? "All answers with a problem" : formatCategoryLabel(selectedCategory)}
                <span className="ml-2 font-normal text-ink-soft">{filteredDiagnostics.length}</span>
              </h3>
              {selectedCategory !== "failing" && (
                <button type="button" onClick={() => setSelectedCategory("failing")} className="text-sm text-ink-soft hover:text-ink">
                  Show all causes
                </button>
              )}
            </div>
            <ul className="-mx-6 mt-3 divide-y divide-line border-y border-line">
              {filteredDiagnostics.slice(0, visibleCount).map((d, i) => (
                <li key={`${d.runId}-${d.test_case_id}-${i}`}>
                  <Link
                    to={`/runs/${d.runId}/test-cases/${d.test_case_id}`}
                    className="flex flex-col gap-1 px-6 py-3 text-sm hover:bg-paper-dim sm:flex-row sm:items-center sm:justify-between"
                  >
                    <span className="min-w-0">
                      <span className="block truncate font-mono text-xs">{d.test_case_id}</span>
                      <span className="text-xs text-ink-soft">
                        Run of {runStarted.has(d.runId) ? formatTimestamp(runStarted.get(d.runId)!) : d.runId.slice(0, 10)}
                      </span>
                    </span>
                    <span className="shrink-0 text-ink-soft">
                      {d.failure_category ? formatCategoryLabel(d.failure_category) : "—"} →
                    </span>
                  </Link>
                </li>
              ))}
            </ul>
            {filteredDiagnostics.length > visibleCount && (
              <button
                type="button"
                onClick={() => setVisibleCount((n) => n + 30)}
                className="mt-4 text-sm text-ink-soft hover:text-ink"
              >
                Show more ({filteredDiagnostics.length - visibleCount} left)
              </button>
            )}
          </section>
        </div>
      )}
    </DataStateView>
  );
}
