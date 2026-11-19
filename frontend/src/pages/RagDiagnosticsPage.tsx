import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { FixtureNotice } from "../components/FixtureNotice";
import { Card, SectionHeading, Stat } from "../components/Card";
import { EmptyState } from "../components/EmptyState";
import { Tag } from "../components/Tag";
import { toneForFailureCategory } from "../lib/tone";
import { formatScore, isRagExecution, shortFingerprint } from "../lib/derived";
import { useConnection } from "../context/ConnectionContext";
import { useRuns } from "../data/useData";
import * as fixtures from "../data/dataAccess";
import { DataStateView } from "../components/DataStateView";
import type { FailureCategory, ReliabilityDiagnostic, RetrievalEvaluationResult } from "../types/domain";

export function RagDiagnosticsPage() {
  const { dataMode, client } = useConnection();
  const { data: runs, loading: runsLoading, error: runsError, refetch } = useRuns();

  interface RagItem {
    runId: string;
    testCaseId: string;
    input: string;
    hitScore: number | null;
    mrrScore: number | null;
    recallScore: number | null;
    category: FailureCategory | null;
    secondaryCount: number;
    hasContext: boolean;
  }

  const [items, setItems] = useState<RagItem[]>([]);
  const [loadingDetails, setLoadingDetails] = useState<boolean>(false);
  const [detailError, setDetailError] = useState<Error | null>(null);

  useEffect(() => {
    async function loadData() {
      if (!runs) return;

      if (dataMode === "fixture") {
        const fixtureItems: RagItem[] = [];
        for (const run of runs) {
          if (!run.result) continue;
          const diagnostics = fixtures.getDiagnosticsForRun(run.run_id);
          const retrievalMap = fixtures.getRetrievalResultsForRun(run.run_id);

          for (const tc of run.result.test_case_results) {
            if (!isRagExecution(tc)) continue;
            const retResults = retrievalMap[tc.test_case_id] ?? [];
            const diag = diagnostics[tc.test_case_id];

            let hit: number | null = null;
            let mrr: number | null = null;
            let recall: number | null = null;

            for (const r of retResults) {
              if (r.status === "success" && r.score !== null) {
                if (r.metric_name === "hit_at_k") hit = r.score;
                else if (r.metric_name === "mrr") mrr = r.score;
                else if (r.metric_name === "recall_at_k") recall = r.score;
              }
            }

            fixtureItems.push({
              runId: run.run_id,
              testCaseId: tc.test_case_id,
              input: tc.input_text,
              hitScore: hit,
              mrrScore: mrr,
              recallScore: recall,
              category: diag?.failure_category ?? null,
              secondaryCount: diag?.secondary_findings?.length ?? 0,
              hasContext: Boolean(diag?.retrieved_context?.length),
            });
          }
        }
        setItems(fixtureItems);
        return;
      }

      // Live mode
      setLoadingDetails(true);
      setDetailError(null);
      try {
        const liveItems: RagItem[] = [];
        for (const run of runs) {
          if (!run.result) continue;
          const ragCases = run.result.test_case_results.filter(isRagExecution);
          if (ragCases.length === 0) continue;

          // Fetch retrieval results and diagnostics for this run
          const [retRes, diagRes] = await Promise.all([
            client.listRetrievalResults({ runId: run.run_id, limit: 100 }),
            client.listDiagnostics({ runId: run.run_id, limit: 100 }),
          ]);

          const retMap: Record<string, RetrievalEvaluationResult[]> = {};
          for (const item of retRes.items) {
            retMap[item.test_case_id] = item.retrieval_evaluations;
          }

          const diagMap: Record<string, ReliabilityDiagnostic> = {};
          for (const item of diagRes.items) {
            diagMap[item.test_case_id] = item;
          }

          for (const tc of ragCases) {
            const retResults = retMap[tc.test_case_id] ?? [];
            const diag = diagMap[tc.test_case_id];

            let hit: number | null = null;
            let mrr: number | null = null;
            let recall: number | null = null;

            for (const r of retResults) {
              if (r.status === "success" && r.score !== null) {
                if (r.metric_name === "hit_at_k") hit = r.score;
                else if (r.metric_name === "mrr") mrr = r.score;
                else if (r.metric_name === "recall_at_k") recall = r.score;
              }
            }

            liveItems.push({
              runId: run.run_id,
              testCaseId: tc.test_case_id,
              input: tc.input_text,
              hitScore: hit,
              mrrScore: mrr,
              recallScore: recall,
              category: diag?.failure_category ?? null,
              secondaryCount: diag?.secondary_findings?.length ?? 0,
              hasContext: Boolean(diag?.retrieved_context?.length),
            });
          }
        }
        setItems(liveItems);
      } catch (err: unknown) {
        setDetailError(err instanceof Error ? err : new Error(String(err)));
      } finally {
        setLoadingDetails(false);
      }
    }

    void loadData();
  }, [runs, dataMode, client]);

  const hitScores = items.map((i) => i.hitScore).filter((s): s is number => s !== null);
  const mrrScores = items.map((i) => i.mrrScore).filter((s): s is number => s !== null);
  const recallScores = items.map((i) => i.recallScore).filter((s): s is number => s !== null);

  const categoryCounts = new Map<FailureCategory, number>();
  for (const item of items) {
    if (item.category) {
      categoryCounts.set(item.category, (categoryCounts.get(item.category) ?? 0) + 1);
    }
  }

  const mean = (arr: number[]) =>
    arr.length > 0 ? (arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(2) : "n/a";

  const loading = runsLoading || loadingDetails;
  const error = runsError || detailError;

  return (
    <div>
      <SectionHeading eyebrow="Retrieval & groundedness" title="RAG Diagnostics" />
      <FixtureNotice />

      <DataStateView loading={loading} error={error} onRetry={refetch}>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat label="RAG Test Cases" value={items.length} />
          <Stat label="Mean Hit@1" value={mean(hitScores)} />
          <Stat label="Mean MRR" value={mean(mrrScores)} />
          <Stat label="Mean Recall@1" value={mean(recallScores)} />
        </div>

        <div className="mt-8 grid gap-6 lg:grid-cols-2">
          <Card>
            <SectionHeading eyebrow="Attribution" title="RAG Failure Attribution" />
            <p className="mb-4 text-xs text-ink-soft">
              The platform decouples retrieval quality from generation faithfulness. A retrieval
              miss and an incorrect answer indicate a <span className="font-mono">retrieval_failure</span>;
              a retrieval hit and an incorrect answer indicate a <span className="font-mono">generation_failure</span>.
            </p>
            {categoryCounts.size === 0 ? (
              <EmptyState title="No RAG failure categories recorded." />
            ) : (
              <div className="space-y-2">
                {[...categoryCounts.entries()].map(([cat, count]) => {
                  const max = Math.max(...categoryCounts.values());
                  return (
                    <div key={cat} className="flex items-center gap-3 text-sm">
                      <span className="w-48 shrink-0">
                        <Tag tone={toneForFailureCategory(cat)}>{cat}</Tag>
                      </span>
                      <span className="h-3 flex-1 bg-paper-dim">
                        <span
                          className="block h-3 bg-ink"
                          style={{ width: `${(count / max) * 100}%` }}
                        />
                      </span>
                      <span className="w-8 text-right font-mono text-xs text-ink-soft">{count}</span>
                    </div>
                  );
                })}
              </div>
            )}
          </Card>

          <Card>
            <SectionHeading eyebrow="Methodology note" title="Faithfulness Assessment Scope" />
            <div className="space-y-3 text-xs text-ink-soft leading-relaxed">
              <p>
                <strong className="text-ink">Lexical overlap baseline:</strong> checks literal token
                intersection between generated text and retrieved passages. It cannot detect negation
                or semantic contradictions, and cannot verify truth outside token matching.
              </p>
              <p>
                <strong className="text-ink">LLM-as-a-judge:</strong> uses a prompt rubric to identify
                supported, unsupported, or contradicted claims. It is subject to position bias,
                verbosity bias, and judge hallucinations.
              </p>
              <p className="border-t border-line pt-2 italic text-ink">
                Neither approach constitutes a universal hallucination detector; both serve as
                inspectable, evidence-gathering signals.
              </p>
            </div>
          </Card>
        </div>

        <div className="mt-8">
          <Card>
            <SectionHeading eyebrow="Investigation" title="RAG Test Case Registry" />
            {items.length === 0 ? (
              <EmptyState title="No RAG executions recorded." />
            ) : (
              <div className="overflow-x-auto border border-line">
                <table className="w-full min-w-[800px] border-collapse text-sm">
                  <thead>
                    <tr className="border-b border-line bg-paper-dim text-left font-mono text-xs uppercase">
                      <th className="px-3 py-2">Test Case ID</th>
                      <th className="px-3 py-2">Run</th>
                      <th className="px-3 py-2">Hit@1</th>
                      <th className="px-3 py-2">MRR</th>
                      <th className="px-3 py-2">Recall@1</th>
                      <th className="px-3 py-2">Failure Category</th>
                      <th className="px-3 py-2">Secondary Findings</th>
                      <th className="px-3 py-2 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {items.map((item, idx) => (
                      <tr
                        key={idx}
                        className="border-b border-line hover:bg-paper-dim/40 last:border-0"
                      >
                        <td className="px-3 py-2 font-mono text-xs font-medium">
                          <Link
                            to={`/runs/${item.runId}/test-cases/${item.testCaseId}`}
                            className="underline"
                          >
                            {item.testCaseId}
                          </Link>
                        </td>
                        <td className="px-3 py-2 font-mono text-xs text-ink-soft">
                          <Link to={`/runs/${item.runId}`} className="underline">
                            {shortFingerprint(item.runId, 12)}
                          </Link>
                        </td>
                        <td className="px-3 py-2 font-mono text-xs">{formatScore(item.hitScore)}</td>
                        <td className="px-3 py-2 font-mono text-xs">{formatScore(item.mrrScore)}</td>
                        <td className="px-3 py-2 font-mono text-xs">{formatScore(item.recallScore)}</td>
                        <td className="px-3 py-2">
                          {item.category ? (
                            <Tag tone={toneForFailureCategory(item.category)}>{item.category}</Tag>
                          ) : (
                            <span className="font-mono text-xs text-ink-soft">none</span>
                          )}
                        </td>
                        <td className="px-3 py-2">
                          {item.secondaryCount > 0 ? (
                            <Tag tone="caution">{`${item.secondaryCount} secondary`}</Tag>
                          ) : (
                            <span className="text-xs text-ink-soft">none</span>
                          )}
                        </td>
                        <td className="px-3 py-2 text-right">
                          <Link
                            to={`/runs/${item.runId}/test-cases/${item.testCaseId}`}
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
            )}
          </Card>
        </div>
      </DataStateView>
    </div>
  );
}
