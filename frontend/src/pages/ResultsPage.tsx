import { useState, type ReactNode } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { FixtureNotice } from "../components/FixtureNotice";
import { PageHeader } from "../components/Card";
import { EmptyState } from "../components/EmptyState";
import { Tag } from "../components/Tag";
import { DataStateView } from "../components/DataStateView";
import { VERDICT_HELP, VerdictBar, VerdictTag, verdictSentence } from "../components/Verdict";
import { toneForClaimSupport } from "../lib/tone";
import {
  CATEGORY_HELP,
  VERDICT_LABEL,
  evaluatorHelp,
  evaluatorLabel,
  evaluatorVerdict,
  formatCategoryLabel,
  formatScore,
  formatTimestamp,
  getRagMetadata,
  testCaseVerdict,
  verdictCounts,
  type Verdict,
} from "../lib/derived";
import {
  useDiagnostic,
  useDiagnostics,
  useExperiments,
  useRetrievalResults,
  useRun,
  useRuns,
} from "../data/useData";
import type { TestCaseResult } from "../types/domain";

type Filter = "all" | Verdict;

function Disclosure({ title, hint, children }: { title: string; hint?: string; children: ReactNode }) {
  return (
    <details className="group rounded-xl border border-line">
      <summary className="flex cursor-pointer select-none items-center justify-between gap-3 px-4 py-3 text-sm hover:bg-paper-dim">
        <span className="flex items-center gap-2">
          <span className="text-ink-soft transition-transform group-open:rotate-90">›</span>
          {title}
        </span>
        {hint && <span className="text-xs text-ink-soft">{hint}</span>}
      </summary>
      <div className="border-t border-line p-4 text-sm">{children}</div>
    </details>
  );
}

function answerText(tc: TestCaseResult): string | null {
  return tc.model_response?.output_text ?? tc.output_text ?? null;
}

export function ResultsPage() {
  const { runId: routeRunId, testCaseId: routeTestCaseId } = useParams<{
    runId?: string;
    testCaseId?: string;
  }>();
  const navigate = useNavigate();

  const { data: allRuns, loading: runsLoading, error: runsError } = useRuns();
  const { data: experiments } = useExperiments();
  const runList = [...(allRuns ?? [])].sort((a, b) => b.started_at.localeCompare(a.started_at));
  const nameOf = new Map((experiments ?? []).map((e) => [e.experiment_id, e.name]));

  const activeRunId = routeRunId || (runList.length > 0 ? runList[0].run_id : "");
  const { data: run, loading: runLoading, error: runError, refetch: refetchRun } = useRun(activeRunId);
  const { data: diagnosticsMap } = useDiagnostics(activeRunId);
  const { data: retrievalMap } = useRetrievalResults(activeRunId);

  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(routeTestCaseId || null);
  const [filter, setFilter] = useState<Filter>("all");
  const [search, setSearch] = useState("");

  const testCases: TestCaseResult[] = run?.result?.test_case_results ?? [];
  const counts = verdictCounts(testCases);
  const diagnostics = diagnosticsMap ?? {};

  const visibleCases = testCases.filter((tc) => {
    if (filter !== "all" && testCaseVerdict(tc) !== filter) return false;
    const q = search.trim().toLowerCase();
    if (q && !tc.test_case_id.toLowerCase().includes(q) && !tc.input_text.toLowerCase().includes(q)) {
      return false;
    }
    return true;
  });

  const activeCaseId = selectedCaseId || routeTestCaseId || testCases[0]?.test_case_id || "";
  const current = testCases.find((tc) => tc.test_case_id === activeCaseId);
  const { data: activeDiag } = useDiagnostic(activeRunId, activeCaseId);
  const ragMeta = current ? getRagMetadata(current.model_response) : null;
  const retrievalScores = retrievalMap?.[activeCaseId] ?? [];

  const loading = (runsLoading && runList.length === 0) || (runLoading && !run);

  // Failure reasons for this run, from the diagnostics service.
  const reasons = new Map<string, number>();
  for (const tc of testCases) {
    const cat = diagnostics[tc.test_case_id]?.failure_category;
    if (cat && cat !== "correct" && testCaseVerdict(tc) !== "passed") {
      reasons.set(cat, (reasons.get(cat) ?? 0) + 1);
    }
  }

  const filters: Filter[] = ["all", "passed", "failed", "unscored", "error"];

  return (
    <div>
      <PageHeader
        title="Results"
        description="Pick a run to see how the model did on each question, and why the wrong answers went wrong."
        action={
          runList.length > 0 && (
            <label className="flex items-center gap-2 text-sm text-ink-soft">
              Run
              <select
                id="run-select"
                value={activeRunId}
                onChange={(e) => {
                  setSelectedCaseId(null);
                  navigate(`/results/${e.target.value}`);
                }}
                className="max-w-[18rem] rounded-md border border-line bg-paper px-3 py-1.5 text-sm text-ink focus:border-ink focus:outline-none"
              >
                {runList.map((r) => (
                  <option key={r.run_id} value={r.run_id}>
                    {nameOf.get(r.experiment_id) ?? r.experiment_id} · {formatTimestamp(r.started_at)}
                  </option>
                ))}
              </select>
            </label>
          )
        }
      />

      <FixtureNotice />

      <DataStateView loading={loading} error={runError || runsError} onRetry={refetchRun}>
        {runList.length === 0 ? (
          <div className="space-y-4">
            <EmptyState title="No runs yet" detail="Run an experiment to see its results here." />
            <Link to="/experiments" className="inline-block rounded-md bg-ink px-4 py-2 text-sm font-medium text-paper">
              Go to Experiments
            </Link>
          </div>
        ) : !run || !run.result ? (
          <EmptyState title="Run not found" detail={`There is no run with ID "${activeRunId}".`} />
        ) : (
          <div className="space-y-8">
            {/* Run summary */}
            <section className="grid gap-6 rounded-xl border border-line bg-paper-dim p-6 lg:grid-cols-5">
              <div className="min-w-0 lg:col-span-3">
                <p className="text-sm text-ink-soft">{formatTimestamp(run.started_at)}</p>
                <h2 className="mt-1 text-xl font-semibold">{nameOf.get(run.experiment_id) ?? run.experiment_id}</h2>
                <p className="mt-1 text-sm text-ink-soft">
                  Model <span className="font-mono text-ink">{run.model_config?.model_id}</span> ·{" "}
                  {counts.total} questions
                  {run.duration_seconds != null && ` · took ${run.duration_seconds.toFixed(1)}s`}
                </p>
                <p className="mt-5 text-base">{verdictSentence(counts)}</p>
                <div className="mt-3 max-w-xl">
                  <VerdictBar counts={counts} />
                </div>
              </div>
              <div className="lg:col-span-2 lg:border-l lg:border-line lg:pl-6">
                <p className="text-sm font-medium">Why answers didn't pass</p>
                {reasons.size === 0 ? (
                  <p className="mt-2 text-sm text-ink-soft">
                    {counts.passed === counts.total
                      ? "Every answer passed."
                      : "No specific cause was diagnosed. Open a question below for details."}
                  </p>
                ) : (
                  <ul className="mt-3 space-y-3">
                    {[...reasons.entries()]
                      .sort((a, b) => b[1] - a[1])
                      .map(([cat, n]) => (
                        <li key={cat}>
                          <p className="flex justify-between text-sm">
                            <span>{formatCategoryLabel(cat)}</span>
                            <span className="tabular-nums text-ink-soft">{n}</span>
                          </p>
                          <p className="text-xs text-ink-soft">{CATEGORY_HELP[cat] ?? ""}</p>
                        </li>
                      ))}
                  </ul>
                )}
              </div>
            </section>

            {/* Questions */}
            <section className="grid gap-6 lg:grid-cols-12">
              <div className="min-w-0 lg:col-span-5">
                <div className="mb-3 flex flex-wrap gap-1">
                  {filters
                    .filter((f) => f === "all" || counts[f] > 0)
                    .map((f) => (
                      <button
                        key={f}
                        type="button"
                        onClick={() => setFilter(f)}
                        title={f === "all" ? undefined : VERDICT_HELP[f]}
                        className={`rounded-md px-2.5 py-1 text-xs transition-colors ${
                          filter === f ? "bg-paper-raised font-medium text-ink" : "text-ink-soft hover:text-ink"
                        }`}
                      >
                        {f === "all" ? "All" : VERDICT_LABEL[f]}{" "}
                        <span className="tabular-nums">{f === "all" ? counts.total : counts[f]}</span>
                      </button>
                    ))}
                </div>
                <input
                  type="search"
                  placeholder="Search questions"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className="mb-3 w-full rounded-md border border-line bg-paper px-3 py-1.5 text-sm placeholder:text-ink-soft focus:border-ink focus:outline-none"
                />
                <ul className="max-h-[640px] divide-y divide-line overflow-y-auto rounded-xl border border-line">
                  {visibleCases.length === 0 ? (
                    <li className="p-4 text-center text-sm text-ink-soft">No questions match.</li>
                  ) : (
                    visibleCases.map((tc) => {
                      const selected = tc.test_case_id === activeCaseId;
                      const cat = diagnostics[tc.test_case_id]?.failure_category;
                      return (
                        <li key={tc.test_case_id}>
                          <button
                            type="button"
                            onClick={() => setSelectedCaseId(tc.test_case_id)}
                            className={`w-full px-4 py-3 text-left transition-colors ${
                              selected ? "bg-paper-raised" : "hover:bg-paper-dim"
                            }`}
                          >
                            <div className="flex items-start justify-between gap-3">
                              <p className="line-clamp-2 text-sm">{tc.input_text}</p>
                              <VerdictTag verdict={testCaseVerdict(tc)} />
                            </div>
                            <p className="mt-1 truncate font-mono text-xs text-ink-soft">
                              {tc.test_case_id}
                              {cat && cat !== "correct" && testCaseVerdict(tc) !== "passed" && (
                                <span className="font-sans"> · {formatCategoryLabel(cat)}</span>
                              )}
                            </p>
                          </button>
                        </li>
                      );
                    })
                  )}
                </ul>
              </div>

              <div className="min-w-0 lg:col-span-7">
                {!current ? (
                  <EmptyState title="Select a question to see the details." />
                ) : (
                  <div className="space-y-4">
                    <div className="rounded-xl border border-line p-5">
                      <div className="flex items-start justify-between gap-3">
                        <p className="font-mono text-xs text-ink-soft">{current.test_case_id}</p>
                        <VerdictTag verdict={testCaseVerdict(current)} />
                      </div>
                      <p className="mt-2 text-lg font-medium">{current.input_text}</p>
                      <p className="mt-1 text-xs text-ink-soft">{VERDICT_HELP[testCaseVerdict(current)]}</p>

                      <div className="mt-5 grid gap-3 sm:grid-cols-2">
                        <div className="rounded-lg bg-paper-dim p-4">
                          <p className="text-xs text-ink-soft">Expected answer</p>
                          <p className="mt-1 text-sm">
                            {current.reference_answer ?? (
                              <span className="text-ink-soft italic">None given</span>
                            )}
                          </p>
                        </div>
                        <div className="rounded-lg bg-paper-dim p-4">
                          <p className="text-xs text-ink-soft">Model's answer</p>
                          <p className="mt-1 text-sm">
                            {current.execution_error ? (
                              <span className="text-accent">{current.execution_error}</span>
                            ) : (
                              answerText(current) ?? <span className="text-ink-soft italic">Empty</span>
                            )}
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="rounded-xl border border-line p-5">
                      <p className="text-sm font-medium">What the checks said</p>
                      {current.evaluation_results.length === 0 ? (
                        <p className="mt-2 text-sm text-ink-soft">No checks were configured for this run.</p>
                      ) : (
                        <ul className="mt-3 divide-y divide-line">
                          {current.evaluation_results.map((e, i) => {
                            const v = evaluatorVerdict(e);
                            return (
                              <li key={i} className="py-3 first:pt-0 last:pb-0">
                                <div className="flex items-center justify-between gap-3">
                                  <p className="text-sm">{evaluatorLabel(e.evaluator_name)}</p>
                                  <Tag tone={v === "passed" ? "neutral" : v === "failed" ? "negative" : "muted"}>
                                    {v === "passed" ? "Passed" : v === "failed" ? "Wrong" : "Couldn't run"}
                                  </Tag>
                                </div>
                                {evaluatorHelp(e.evaluator_name) && (
                                  <p className="text-xs text-ink-soft">{evaluatorHelp(e.evaluator_name)}</p>
                                )}
                                {(e.explanation || e.error) && (
                                  <p className={`mt-1.5 text-xs ${e.error ? "text-accent" : "text-ink-soft"}`}>
                                    {e.error ? `Error: ${e.error}` : e.explanation}
                                  </p>
                                )}
                              </li>
                            );
                          })}
                        </ul>
                      )}
                    </div>

                    {activeDiag?.failure_category && (
                      <div className="rounded-xl border border-line p-5">
                        <div className="flex items-center justify-between gap-3">
                          <p className="text-sm font-medium">Diagnosis</p>
                          <span className="text-sm">{formatCategoryLabel(activeDiag.failure_category)}</span>
                        </div>
                        <p className="mt-2 text-sm text-ink-soft">
                          {activeDiag.explanation || CATEGORY_HELP[activeDiag.failure_category]}
                        </p>
                        {activeDiag.secondary_findings?.length > 0 && (
                          <ul className="mt-2 list-disc pl-5 text-xs text-ink-soft">
                            {activeDiag.secondary_findings.map((f, i) => (
                              <li key={i}>{f}</li>
                            ))}
                          </ul>
                        )}
                      </div>
                    )}

                    {ragMeta && (
                      <Disclosure
                        title={`Passages the model was given (${ragMeta.retrieved_chunks.length})`}
                        hint="Retrieval"
                      >
                        <ol className="space-y-3">
                          {ragMeta.retrieved_chunks.map((c) => (
                            <li key={c.chunk_id} className="rounded-lg bg-paper-dim p-3">
                              <p className="text-xs text-ink-soft">
                                #{c.rank} · similarity {c.score.toFixed(3)} ·{" "}
                                <span className="font-mono">{c.chunk_id}</span>
                              </p>
                              <p className="mt-1">{c.text}</p>
                            </li>
                          ))}
                        </ol>
                        {retrievalScores.length > 0 && (
                          <p className="mt-3 text-xs text-ink-soft">
                            {retrievalScores.map((r) => `${r.metric_name} ${formatScore(r.score)}`).join(" · ")}
                          </p>
                        )}
                      </Disclosure>
                    )}

                    {activeDiag?.claim_assessments && activeDiag.claim_assessments.length > 0 && (
                      <Disclosure
                        title={`Sentence-by-sentence support (${activeDiag.claim_assessments.length})`}
                        hint="Faithfulness"
                      >
                        <ul className="space-y-3">
                          {activeDiag.claim_assessments.map((ca, i) => {
                            const status = ca.status ?? ca.assessment?.status ?? "undetermined";
                            const why = ca.reasoning ?? ca.assessment?.explanation ?? ca.evidence_snippets?.join("; ");
                            return (
                              <li key={i}>
                                <div className="flex items-start justify-between gap-3">
                                  <p>{ca.statement}</p>
                                  <Tag tone={toneForClaimSupport(status)}>{status}</Tag>
                                </div>
                                {why && <p className="mt-1 text-xs text-ink-soft">{why}</p>}
                              </li>
                            );
                          })}
                        </ul>
                      </Disclosure>
                    )}

                    <Disclosure
                      title="Speed and token usage"
                      hint={
                        current.model_response?.latency_ms != null
                          ? `${current.model_response.latency_ms.toFixed(0)} ms`
                          : undefined
                      }
                    >
                      <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                        {[
                          ["Latency", current.model_response?.latency_ms != null ? `${current.model_response.latency_ms.toFixed(1)} ms` : "n/a"],
                          ["Prompt tokens", current.model_response?.prompt_tokens ?? "n/a"],
                          ["Answer tokens", current.model_response?.completion_tokens ?? "n/a"],
                          ["Total tokens", current.model_response?.total_tokens ?? "n/a"],
                        ].map(([k, v]) => (
                          <div key={String(k)}>
                            <dt className="text-xs text-ink-soft">{k}</dt>
                            <dd className="tabular-nums">{v}</dd>
                          </div>
                        ))}
                      </dl>
                    </Disclosure>
                  </div>
                )}
              </div>
            </section>
          </div>
        )}
      </DataStateView>
    </div>
  );
}
