import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { FixtureNotice } from "../components/FixtureNotice";
import { Card, SectionHeading } from "../components/Card";
import { EmptyState } from "../components/EmptyState";
import { Tag } from "../components/Tag";
import {
  toneForClaimSupport,
  toneForFailureCategory,
  toneForRunStatus,
} from "../lib/tone";
import {
  formatCategoryLabel,
  formatScore,
  formatTimestamp,
  getRagMetadata,
  shortFingerprint,
  testCasePassed,
  testCaseSucceeded,
} from "../lib/derived";
import {
  useDiagnostic,
  useDiagnostics,
  useExperiment,
  useRetrievalResults,
  useRun,
  useRuns,
} from "../data/useData";
import { DataStateView } from "../components/DataStateView";
import { PassFailDonutChart } from "../components/charts/PassFailDonutChart";
import { FailureDistributionChart } from "../components/charts/FailureDistributionChart";
import type { FailureCategory, TestCaseResult } from "../types/domain";

export function ResultsPage() {
  const { runId: routeRunId, testCaseId: routeTestCaseId } = useParams<{
    runId?: string;
    testCaseId?: string;
  }>();
  const navigate = useNavigate();

  // Load all runs for the run-selector
  const { data: allRuns, loading: runsLoading, error: runsError } = useRuns();
  const runList = allRuns || [];

  // Determine active run ID: route param, or fallback to latest run
  const activeRunId = routeRunId || (runList.length > 0 ? runList[0].run_id : "");

  // Load details for the active run
  const {
    data: run,
    loading: runLoading,
    error: runError,
    refetch: refetchRun,
  } = useRun(activeRunId);
  const { data: experiment } = useExperiment(run?.experiment_id || "");
  const { data: diagnosticsMap, loading: diagLoading } = useDiagnostics(activeRunId);

  // Active selected test case for forensic inspection
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(routeTestCaseId || null);
  const [categoryFilter, setCategoryFilter] = useState<string>("all");
  const [statusFilter, setStatusFilter] = useState<"all" | "passed" | "failed" | "error">("all");
  const [caseSearchInput, setCaseSearchInput] = useState<string>("");

  const testCases: TestCaseResult[] = run?.result?.test_case_results ?? [];

  // Load specific diagnostics and retrieval if a case is selected
  const activeCaseId =
    selectedCaseId || routeTestCaseId || (testCases.length > 0 ? testCases[0].test_case_id : "");
  const { data: activeDiag } = useDiagnostic(activeRunId, activeCaseId);
  const { data: retrievalMap } = useRetrievalResults(activeRunId);

  const loading = (runsLoading && runList.length === 0) || (runLoading && !run) || diagLoading;
  const diagnostics = diagnosticsMap || {};

  // Summary counts - strictly separating execution health from evaluation correctness
  const totalCases = testCases.length;
  const passedCount = testCases.filter(testCasePassed).length;
  const evalFailedCount = totalCases - passedCount;

  const execSucceededCount = testCases.filter(testCaseSucceeded).length;
  const execFailedCount = totalCases - execSucceededCount;
  const execSuccessRate = totalCases > 0 ? ((execSucceededCount / totalCases) * 100).toFixed(1) : null;

  // Category counts for filter & visual chart
  const categoryCounts = new Map<FailureCategory, number>();
  for (const tc of testCases) {
    const diag = diagnostics[tc.test_case_id];
    if (diag?.failure_category) {
      categoryCounts.set(
        diag.failure_category,
        (categoryCounts.get(diag.failure_category) ?? 0) + 1
      );
    } else if (testCaseSucceeded(tc) && !testCasePassed(tc)) {
      // Fallback categorization if diagnostic not yet resolved
      categoryCounts.set("incorrect", (categoryCounts.get("incorrect") ?? 0) + 1);
    }
  }

  // Filtered test cases
  const filteredCases = testCases.filter((tc) => {
    const passed = testCasePassed(tc);
    const succeeded = testCaseSucceeded(tc);
    if (statusFilter === "passed" && !passed) return false;
    if (statusFilter === "failed" && (passed || !succeeded)) return false;
    if (statusFilter === "error" && succeeded) return false;

    if (categoryFilter !== "all") {
      const diag = diagnostics[tc.test_case_id];
      const actualCat = diag?.failure_category ?? (succeeded && !passed ? "incorrect" : null);
      if (actualCat !== categoryFilter) return false;
    }

    if (caseSearchInput.trim()) {
      const q = caseSearchInput.toLowerCase();
      const matchId = tc.test_case_id.toLowerCase().includes(q);
      const matchText = tc.input_text.toLowerCase().includes(q);
      if (!matchId && !matchText) return false;
    }

    return true;
  });

  // Selected test case result
  const currentCaseResult = testCases.find((tc) => tc.test_case_id === activeCaseId);
  const currentRagMeta = currentCaseResult ? getRagMetadata(currentCaseResult.model_response) : null;
  const currentRetrievalList = (retrievalMap && retrievalMap[activeCaseId]) || [];

  // Handle run selection switch
  const handleSelectRun = (newRunId: string) => {
    setSelectedCaseId(null);
    navigate(`/results/${newRunId}`);
  };

  return (
    <div className="space-y-6">
      {/* Header and Run Selector */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between border-b border-line pb-4">
        <div>
          <SectionHeading eyebrow="Evaluation Results" title="Run Outcomes & Forensics" />
          <p className="mt-1 text-xs text-ink-soft">
            Visual performance summary, failure mode attribution, and focused test-case forensic evidence.
          </p>
        </div>

        {/* Run Selector Dropdown */}
        {runList.length > 0 && (
          <div className="flex items-center gap-2">
            <label htmlFor="run-select" className="font-mono text-xs text-ink-soft">
              Select Run:
            </label>
            <select
              id="run-select"
              value={activeRunId}
              onChange={(e) => handleSelectRun(e.target.value)}
              className="border border-line bg-paper px-3 py-1.5 font-mono text-xs text-ink focus:border-ink focus:outline-none"
            >
              {runList.map((r) => (
                <option key={r.run_id} value={r.run_id}>
                  {shortFingerprint(r.run_id, 12)} — {r.model_config?.model_id || "model"} ({r.status})
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      <FixtureNotice />

      <DataStateView loading={loading} error={runError || runsError} onRetry={refetchRun}>
        {runList.length === 0 ? (
          <div>
            <EmptyState
              title="No evaluation runs have been executed yet."
              detail="To see results, navigate to Experiments and execute an evaluation run."
            />
            <Link
              to="/experiments"
              className="mt-3 inline-block bg-ink px-4 py-2 font-mono text-xs font-medium text-paper hover:bg-ink-soft"
            >
              Go to Experiments &rarr;
            </Link>
          </div>
        ) : !run || !run.result ? (
          <EmptyState
            title="Run not found"
            detail={`No evaluation run with ID "${activeRunId}" exists.`}
          />
        ) : (
          <>
            {/* Top Run Metadata Bar */}
            <div className="border border-line bg-paper p-4">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[11px] uppercase tracking-wider text-ink-soft">Experiment</span>
                    <Tag tone={toneForRunStatus(run.status)}>{run.status}</Tag>
                  </div>
                  <h3 className="font-display text-lg font-medium text-ink mt-0.5">
                    {experiment ? experiment.name : run.experiment_id}
                  </h3>
                  <p className="font-mono text-xs text-ink-soft">
                    Run ID: {run.run_id} &bull; Started: {formatTimestamp(run.started_at)}
                  </p>
                </div>

                <div className="flex items-center gap-4 text-xs font-mono">
                  <div className="border border-line px-3 py-1.5 bg-paper-dim">
                    <span className="text-ink-soft uppercase text-[10px] block">Model</span>
                    <span className="text-ink font-semibold">{run.model_config?.model_id || "Unspecified"}</span>
                  </div>
                  <div className="border border-line px-3 py-1.5 bg-paper-dim">
                    <span className="text-ink-soft uppercase text-[10px] block">Dataset</span>
                    <span className="text-ink">{run.dataset_version?.dataset_name || "N/A"}</span>
                  </div>
                  <div className="border border-line px-3 py-1.5 bg-paper-dim">
                    <span className="text-ink-soft uppercase text-[10px] block">Execution Time</span>
                    <span className="text-ink">
                      {run.duration_seconds !== null && run.duration_seconds !== undefined
                        ? `${run.duration_seconds.toFixed(2)}s`
                        : "n/a"}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* VISUAL DASHBOARD SECTION (2 Visual Cards Side-by-Side) */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* Card 1: Visual Performance Summary (Donut Chart) */}
              <div className="lg:col-span-6 border border-line bg-paper p-5 flex flex-col justify-between">
                <div>
                  <div className="border-b border-line pb-2 mb-4">
                    <h4 className="font-display text-sm font-medium text-ink">
                      Visual Evaluation Performance
                    </h4>
                    <span className="text-[11px] text-ink-soft">
                      Ratio of test cases satisfying evaluator criteria vs failing judgments
                    </span>
                  </div>

                  <div className="py-2 flex justify-center">
                    <PassFailDonutChart
                      passed={passedCount}
                      failed={evalFailedCount - execFailedCount}
                      executionErrors={execFailedCount}
                      total={totalCases}
                      size={180}
                      thickness={22}
                    />
                  </div>
                </div>

                {/* Execution Health Grounding Statement */}
                <div className="mt-4 pt-3 border-t border-line text-xs font-mono flex items-center justify-between text-ink-soft">
                  <span>Execution Completion:</span>
                  <span className="text-ink font-semibold">
                    {execSucceededCount} of {totalCases} completed ({execSuccessRate || 0}%)
                    {execFailedCount === 0 ? " — 0 runtime crashes" : ` — ${execFailedCount} crashed`}
                  </span>
                </div>
              </div>

              {/* Card 2: Failure Category Distribution */}
              <div className="lg:col-span-6">
                <FailureDistributionChart
                  categoryCounts={categoryCounts}
                  totalFailures={evalFailedCount}
                  title="Failure Mode Attribution"
                  emptyMessage="All test cases satisfied evaluator criteria without failure."
                />
              </div>
            </div>

            {/* TEST CASE RESULTS & FOCUSED INVESTIGATION SECTION */}
            <div className="grid gap-6 lg:grid-cols-12">
              {/* Left Column: Clean Test Case Table & Filters (5 cols) */}
              <div className="lg:col-span-5 space-y-3">
                <Card>
                  <div className="flex items-center justify-between border-b border-line pb-3 mb-3">
                    <div>
                      <h4 className="font-display text-base font-semibold">
                        Test Cases ({filteredCases.length})
                      </h4>
                      <span className="text-[11px] text-ink-soft">Click any case to inspect details</span>
                    </div>
                  </div>

                  {/* Filter Controls & Search */}
                  <div className="space-y-2 mb-3 text-xs">
                    <input
                      type="text"
                      placeholder="Search question text or ID..."
                      value={caseSearchInput}
                      onChange={(e) => setCaseSearchInput(e.target.value)}
                      className="w-full border border-line bg-paper px-2.5 py-1 text-xs text-ink placeholder:text-ink-soft focus:border-ink focus:outline-none"
                    />

                    <div className="flex flex-wrap items-center gap-2">
                      <div className="flex items-center border border-line p-0.5 bg-paper-dim">
                        <button
                          type="button"
                          onClick={() => setStatusFilter("all")}
                          className={`px-2 py-0.5 font-mono text-[11px] ${
                            statusFilter === "all" ? "bg-paper font-semibold text-ink" : "text-ink-soft"
                          }`}
                        >
                          All ({totalCases})
                        </button>
                        <button
                          type="button"
                          onClick={() => setStatusFilter("passed")}
                          className={`px-2 py-0.5 font-mono text-[11px] ${
                            statusFilter === "passed" ? "bg-paper font-semibold text-ink" : "text-ink-soft"
                          }`}
                        >
                          Pass ({passedCount})
                        </button>
                        <button
                          type="button"
                          onClick={() => setStatusFilter("failed")}
                          className={`px-2 py-0.5 font-mono text-[11px] ${
                            statusFilter === "failed" ? "bg-paper font-semibold text-ink" : "text-ink-soft"
                          }`}
                        >
                          Fail ({evalFailedCount - execFailedCount})
                        </button>
                        {execFailedCount > 0 && (
                          <button
                            type="button"
                            onClick={() => setStatusFilter("error")}
                            className={`px-2 py-0.5 font-mono text-[11px] ${
                              statusFilter === "error" ? "bg-paper font-semibold text-ink" : "text-ink-soft"
                            }`}
                          >
                            Error ({execFailedCount})
                          </button>
                        )}
                      </div>

                      {categoryCounts.size > 0 && (
                        <select
                          value={categoryFilter}
                          onChange={(e) => setCategoryFilter(e.target.value)}
                          className="border border-line bg-paper px-2 py-0.5 text-xs text-ink focus:outline-none"
                        >
                          <option value="all">All Categories</option>
                          {[...categoryCounts.entries()].map(([cat, count]) => (
                            <option key={cat} value={cat}>
                              {formatCategoryLabel(cat)} ({count})
                            </option>
                          ))}
                        </select>
                      )}
                    </div>
                  </div>

                  {/* Compact Case List */}
                  <div className="divide-y divide-line max-h-[580px] overflow-y-auto border border-line">
                    {filteredCases.length === 0 ? (
                      <div className="p-4 text-center text-xs text-ink-soft">
                        No test cases match the active filter criteria.
                      </div>
                    ) : (
                      filteredCases.map((tc) => {
                        const diag = diagnostics[tc.test_case_id];
                        const succeeded = testCaseSucceeded(tc);
                        const passed = testCasePassed(tc);
                        const isSelected = tc.test_case_id === activeCaseId;
                        const category = diag?.failure_category ?? (succeeded && !passed ? "incorrect" : null);

                        return (
                          <button
                            key={tc.test_case_id}
                            type="button"
                            onClick={() => setSelectedCaseId(tc.test_case_id)}
                            className={`w-full text-left p-3 transition-colors ${
                              isSelected
                                ? "bg-paper-dim border-l-4 border-l-ink font-medium"
                                : "hover:bg-paper-dim"
                            }`}
                          >
                            <div className="flex items-center justify-between text-xs mb-1">
                              <span className="font-mono text-ink-soft">{tc.test_case_id}</span>
                              {!succeeded ? (
                                <Tag tone="negative">Execution Error</Tag>
                              ) : (
                                <Tag tone={passed ? "neutral" : "negative"}>
                                  {passed ? "Passed" : "Failed"}
                                </Tag>
                              )}
                            </div>
                            <p className="text-xs text-ink line-clamp-2 leading-relaxed">
                              {tc.input_text}
                            </p>
                            {category && (
                              <div className="mt-1.5 flex items-center gap-1.5">
                                <span className="font-mono text-[10px] text-ink-soft uppercase">Diagnosis:</span>
                                <span className="font-mono text-[11px] text-ink-soft font-medium">
                                  {formatCategoryLabel(category)}
                                </span>
                              </div>
                            )}
                          </button>
                        );
                      })
                    )}
                  </div>
                </Card>
              </div>

              {/* Right Column: Focused Test-Case Investigation Drawer (7 cols) */}
              <div className="lg:col-span-7">
                {currentCaseResult ? (
                  <div className="space-y-4">
                    {/* Primary Focused Card: Question, Expected, Generated, Outcome */}
                    <Card>
                      <div className="flex items-center justify-between border-b border-line pb-3 mb-4">
                        <div>
                          <SectionHeading
                            eyebrow="Forensic Investigation"
                            title={`Case: ${currentCaseResult.test_case_id}`}
                          />
                        </div>
                        <div className="flex items-center gap-2">
                          <Tag tone={testCaseSucceeded(currentCaseResult) ? "neutral" : "negative"}>
                            {testCaseSucceeded(currentCaseResult) ? "Execution Succeeded" : "Execution Error"}
                          </Tag>
                          <Tag tone={testCasePassed(currentCaseResult) ? "neutral" : "negative"}>
                            {testCasePassed(currentCaseResult) ? "Evaluation Passed" : "Evaluation Failed"}
                          </Tag>
                        </div>
                      </div>

                      {/* Question / Prompt */}
                      <div className="space-y-4 text-xs">
                        <div>
                          <span className="block font-mono text-[11px] uppercase text-ink-soft mb-1 font-semibold">
                            Input Question / Prompt
                          </span>
                          <div className="border border-line bg-paper p-3 font-mono text-sm leading-relaxed text-ink">
                            {currentCaseResult.input_text}
                          </div>
                        </div>

                        {/* VISUAL CONTRAST: Expected Ground Truth vs Generated Answer */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          {/* Expected Ground Truth Answer */}
                          <div className="border-l-4 border-l-emerald-400 border border-line bg-paper-dim p-3 space-y-1">
                            <span className="block font-mono text-[10px] uppercase tracking-wider text-emerald-400 font-bold">
                              ✓ Expected Answer (Ground Truth)
                            </span>
                            <div className="font-mono text-xs text-ink leading-relaxed pt-1">
                              {currentCaseResult.reference_answer || (
                                <span className="text-ink-soft italic">No reference answer specified (reference-free evaluation)</span>
                              )}
                            </div>
                          </div>

                          {/* Model Generated Output */}
                          <div
                            className={`border-l-4 ${
                              testCasePassed(currentCaseResult)
                                ? "border-l-emerald-400"
                                : "border-l-red-400"
                            } border border-line bg-paper-dim p-3 space-y-1`}
                          >
                            <span
                              className={`block font-mono text-[10px] uppercase tracking-wider font-bold ${
                                testCasePassed(currentCaseResult) ? "text-emerald-400" : "text-red-400"
                              }`}
                            >
                              {testCasePassed(currentCaseResult) ? "✓ Generated Output" : "✗ Generated Output (Mismatch)"}
                            </span>
                            <div className="font-mono text-xs text-ink leading-relaxed pt-1">
                              {currentCaseResult.execution_error ? (
                                <span className="text-red-400 font-semibold">{currentCaseResult.execution_error}</span>
                              ) : (
                                currentCaseResult.model_response?.output_text ||
                                currentCaseResult.output_text || (
                                  <span className="text-ink-soft italic">(empty response generated)</span>
                                )
                              )}
                            </div>
                          </div>
                        </div>

                        {/* Plain English Outcome Explanation */}
                        <div className="border border-line bg-paper-dim p-3.5 space-y-1.5">
                          <div className="flex items-center justify-between">
                            <span className="font-mono text-[10px] uppercase text-ink-soft font-semibold">
                              Primary Diagnosis & Explanation
                            </span>
                            {activeDiag?.failure_category && (
                              <Tag tone={toneForFailureCategory(activeDiag.failure_category)}>
                                {formatCategoryLabel(activeDiag.failure_category)}
                              </Tag>
                            )}
                          </div>
                          <p className="text-xs text-ink leading-relaxed">
                            {activeDiag?.explanation ||
                              (testCasePassed(currentCaseResult)
                                ? "The generated answer fully satisfied all configured evaluation criteria."
                                : "The generated answer did not match the reference answer under exact string matching rules.")}
                          </p>

                          {activeDiag?.secondary_findings && activeDiag.secondary_findings.length > 0 && (
                            <div className="mt-2 pt-2 border-t border-line text-[11px] text-ink-soft space-y-0.5">
                              <span className="font-mono font-semibold uppercase text-[10px] block">Secondary Findings:</span>
                              {activeDiag.secondary_findings.map((finding, idx) => (
                                <div key={idx} className="flex items-center gap-1.5">
                                  <span>&bull;</span>
                                  <span>{finding}</span>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                    </Card>

                    {/* PROGRESSIVE DISCLOSURE COLLAPSIBLES (Collapsed by Default) */}
                    <div className="space-y-3">
                      {/* 1. Evaluator Breakdown Accordion */}
                      <details className="group border border-line bg-paper">
                        <summary className="cursor-pointer p-3.5 font-mono text-xs text-ink flex items-center justify-between select-none hover:bg-paper-dim transition-colors">
                          <span className="flex items-center gap-2 font-medium">
                            <span className="text-ink-soft group-open:rotate-90 transition-transform">▸</span>
                            Evaluator Verdicts & Scores ({currentCaseResult.evaluation_results.length})
                          </span>
                          <span className="text-[11px] text-ink-soft">
                            {currentCaseResult.evaluation_results.every((e) => e.passed) ? "All passed" : "Has failures"}
                          </span>
                        </summary>

                        <div className="p-3.5 border-t border-line overflow-x-auto text-xs">
                          {currentCaseResult.evaluation_results.length === 0 ? (
                            <p className="text-ink-soft">No evaluators scored this test case.</p>
                          ) : (
                            <table className="w-full text-left font-mono">
                              <thead>
                                <tr className="border-b border-line bg-paper-dim uppercase text-[11px]">
                                  <th className="p-2">Evaluator</th>
                                  <th className="p-2">Criterion</th>
                                  <th className="p-2">Verdict</th>
                                  <th className="p-2">Score</th>
                                  <th className="p-2">Details / Explanation</th>
                                </tr>
                              </thead>
                              <tbody>
                                {currentCaseResult.evaluation_results.map((e, idx) => (
                                  <tr key={idx} className="border-b border-line last:border-0">
                                    <td className="p-2 font-medium">{e.evaluator_name}</td>
                                    <td className="p-2 text-ink-soft">{e.criterion || "n/a"}</td>
                                    <td className="p-2">
                                      {e.passed === null ? (
                                        <Tag tone="muted">n/a</Tag>
                                      ) : (
                                        <Tag tone={e.passed ? "neutral" : "negative"}>
                                          {e.passed ? "Passed" : "Failed"}
                                        </Tag>
                                      )}
                                    </td>
                                    <td className="p-2">{formatScore(e.score)}</td>
                                    <td className="p-2 text-ink-soft max-w-xs font-sans text-xs">
                                      {e.explanation || e.error || "—"}
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          )}
                        </div>
                      </details>

                      {/* 2. RAG Context Evidence Accordion (if present) */}
                      {currentRagMeta && (
                        <details className="group border border-line bg-paper">
                          <summary className="cursor-pointer p-3.5 font-mono text-xs text-ink flex items-center justify-between select-none hover:bg-paper-dim transition-colors">
                            <span className="flex items-center gap-2 font-medium">
                              <span className="text-ink-soft group-open:rotate-90 transition-transform">▸</span>
                              Retrieved Context Passages ({currentRagMeta.retrieved_chunks.length} chunks)
                            </span>
                            <span className="text-[11px] text-ink-soft">RAG Evidence</span>
                          </summary>

                          <div className="p-3.5 border-t border-line space-y-2 text-xs">
                            {currentRagMeta.retrieved_chunks.map((chunk, idx) => (
                              <div key={idx} className="border border-line p-3 bg-paper-dim space-y-1">
                                <div className="flex items-center justify-between font-mono text-[11px] text-ink-soft">
                                  <span>Chunk ID: {chunk.chunk_id} (Rank #{chunk.rank})</span>
                                  <span>Similarity Score: {chunk.score.toFixed(4)}</span>
                                </div>
                                <p className="font-mono text-ink leading-relaxed">{chunk.text}</p>
                              </div>
                            ))}

                            {currentRetrievalList.length > 0 && (
                              <div className="mt-3 border-t border-line pt-2">
                                <span className="block font-mono text-[10px] uppercase text-ink-soft mb-1">
                                  Retrieval Evaluation Metrics:
                                </span>
                                <div className="flex flex-wrap gap-2">
                                  {currentRetrievalList.map((rm, idx) => (
                                    <div key={idx} className="border border-line px-2 py-1 bg-paper font-mono text-[11px]">
                                      <span className="text-ink-soft">{rm.metric_name}: </span>
                                      <span className="font-semibold">{formatScore(rm.score)}</span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        </details>
                      )}

                      {/* 3. Sentence-Level Claim Grounding Accordion (if present) */}
                      {activeDiag?.claim_assessments && activeDiag.claim_assessments.length > 0 && (
                        <details className="group border border-line bg-paper">
                          <summary className="cursor-pointer p-3.5 font-mono text-xs text-ink flex items-center justify-between select-none hover:bg-paper-dim transition-colors">
                            <span className="flex items-center gap-2 font-medium">
                              <span className="text-ink-soft group-open:rotate-90 transition-transform">▸</span>
                              Sentence-Level Claim Support ({activeDiag.claim_assessments.length} claims)
                            </span>
                            <span className="text-[11px] text-ink-soft">Faithfulness Forensics</span>
                          </summary>

                          <div className="p-3.5 border-t border-line overflow-x-auto text-xs">
                            <table className="w-full text-left font-mono">
                              <thead>
                                <tr className="border-b border-line bg-paper-dim uppercase text-[11px]">
                                  <th className="p-2">Extracted Sentence</th>
                                  <th className="p-2">Support Status</th>
                                  <th className="p-2">Evidence / Explanation</th>
                                </tr>
                              </thead>
                              <tbody>
                                {activeDiag.claim_assessments.map((ca, idx) => {
                                  const claimStatus = ca.status ?? ca.assessment?.status ?? "undetermined";
                                  const reasoning =
                                    ca.reasoning ??
                                    ca.assessment?.explanation ??
                                    (ca.evidence_snippets && ca.evidence_snippets.length > 0
                                      ? ca.evidence_snippets.join("; ")
                                      : "—");
                                  return (
                                    <tr key={idx} className="border-b border-line last:border-0 font-sans">
                                      <td className="p-2 font-mono text-xs">{ca.statement}</td>
                                      <td className="p-2">
                                        <Tag tone={toneForClaimSupport(claimStatus)}>
                                          {claimStatus}
                                        </Tag>
                                      </td>
                                      <td className="p-2 text-ink-soft text-xs">{reasoning}</td>
                                    </tr>
                                  );
                                })}
                              </tbody>
                            </table>
                          </div>
                        </details>
                      )}

                      {/* 4. Execution Telemetry Accordion */}
                      <details className="group border border-line bg-paper">
                        <summary className="cursor-pointer p-3.5 font-mono text-xs text-ink flex items-center justify-between select-none hover:bg-paper-dim transition-colors">
                          <span className="flex items-center gap-2 font-medium">
                            <span className="text-ink-soft group-open:rotate-90 transition-transform">▸</span>
                            Execution Telemetry & Token Usage
                          </span>
                          <span className="text-[11px] text-ink-soft">
                            {currentCaseResult.model_response?.latency_ms !== null &&
                            currentCaseResult.model_response?.latency_ms !== undefined
                              ? `${currentCaseResult.model_response.latency_ms.toFixed(0)} ms`
                              : "n/a"}
                          </span>
                        </summary>

                        <div className="p-3.5 border-t border-line text-xs font-mono grid grid-cols-2 sm:grid-cols-4 gap-3 bg-paper-dim">
                          <div>
                            <span className="text-ink-soft block uppercase text-[10px]">Latency</span>
                            <span className="text-ink font-semibold">
                              {currentCaseResult.model_response?.latency_ms !== null &&
                              currentCaseResult.model_response?.latency_ms !== undefined
                                ? `${currentCaseResult.model_response.latency_ms.toFixed(1)} ms`
                                : "n/a"}
                            </span>
                          </div>
                          <div>
                            <span className="text-ink-soft block uppercase text-[10px]">Prompt Tokens</span>
                            <span className="text-ink">
                              {currentCaseResult.model_response?.prompt_tokens ?? "n/a"}
                            </span>
                          </div>
                          <div>
                            <span className="text-ink-soft block uppercase text-[10px]">Completion Tokens</span>
                            <span className="text-ink">
                              {currentCaseResult.model_response?.completion_tokens ?? "n/a"}
                            </span>
                          </div>
                          <div>
                            <span className="text-ink-soft block uppercase text-[10px]">Total Tokens</span>
                            <span className="text-ink">
                              {currentCaseResult.model_response?.total_tokens ?? "n/a"}
                            </span>
                          </div>
                        </div>
                      </details>
                    </div>
                  </div>
                ) : (
                  <EmptyState title="Select a test case to inspect forensic evidence." />
                )}
              </div>
            </div>
          </>
        )}
      </DataStateView>
    </div>
  );
}
