import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { FixtureNotice } from "../components/FixtureNotice";
import { Card, SectionHeading, Stat } from "../components/Card";
import { EmptyState } from "../components/EmptyState";
import { Tag } from "../components/Tag";
import { formatPercent, formatScore, shortFingerprint } from "../lib/derived";
import { useConnection } from "../context/ConnectionContext";
import { useRuns } from "../data/useData";
import * as fixtures from "../data/dataAccess";
import { DataStateView } from "../components/DataStateView";
import type { EvaluatorAgreementSummary, EvaluatorSummary } from "../types/domain";

export function EvaluatorComparisonPage() {
  const { dataMode, client } = useConnection();
  const { data: runs, loading: runsLoading, error: runsError, refetch } = useRuns();

  interface DisagreementCase {
    runId: string;
    testCaseId: string;
    question: string;
    answer: string;
    results: { name: string; passed: boolean | null; score: number | null; label: string | null }[];
  }

  const [evaluatorAggregates, setEvaluatorAggregates] = useState<Map<string, EvaluatorSummary>>(
    new Map(),
  );
  const [totalDisagreements, setTotalDisagreements] = useState<number>(0);
  const [disagreementCases, setDisagreementCases] = useState<DisagreementCase[]>([]);
  const [agreementsList, setAgreementsList] = useState<EvaluatorAgreementSummary[]>([]);
  const [loadingDetails, setLoadingDetails] = useState<boolean>(false);
  const [detailError, setDetailError] = useState<Error | null>(null);

  useEffect(() => {
    async function loadData() {
      if (!runs) return;

      if (dataMode === "fixture") {
        const aggs = new Map<string, EvaluatorSummary>();
        let disagreements = 0;
        const cases: DisagreementCase[] = [];
        const agreements: EvaluatorAgreementSummary[] = [];

        for (const run of runs) {
          const sumBundle = fixtures.getSummaryForRun(run.run_id);
          if (sumBundle) {
            disagreements += sumBundle.evaluation_summary.disagreement_count;
            for (const [name, ev] of Object.entries(sumBundle.evaluation_summary.per_evaluator)) {
              const existing = aggs.get(name);
              if (!existing) {
                aggs.set(name, { ...ev });
              } else {
                existing.num_evaluated += ev.num_evaluated;
                existing.num_success += ev.num_success;
                existing.num_skipped += ev.num_skipped;
                existing.num_error += ev.num_error;
                existing.num_passed += ev.num_passed;
                existing.num_failed += ev.num_failed;
                existing.score_count += ev.score_count;
                if (ev.score_mean !== null) {
                  existing.score_mean =
                    existing.score_mean !== null
                      ? (existing.score_mean + ev.score_mean) / 2
                      : ev.score_mean;
                }
              }
            }
          }

          const evAnalytics = fixtures.getEvaluatorAnalytics(run.run_id);
          if (evAnalytics) {
            agreements.push(...evAnalytics.agreements);
          }

          if (run.result) {
            for (const tc of run.result.test_case_results) {
              const passedValues = tc.evaluation_results
                .map((e) => e.passed)
                .filter((p): p is boolean => p !== null);
              const uniquePassed = new Set(passedValues);
              if (uniquePassed.size > 1) {
                cases.push({
                  runId: run.run_id,
                  testCaseId: tc.test_case_id,
                  question: tc.input_text,
                  answer: tc.model_response?.output_text || "",
                  results: tc.evaluation_results.map((e) => ({
                    name: e.evaluator_name,
                    passed: e.passed,
                    score: e.score,
                    label: e.label,
                  })),
                });
              }
            }
          }
        }

        setEvaluatorAggregates(aggs);
        setTotalDisagreements(disagreements);
        setDisagreementCases(cases);
        setAgreementsList(agreements);
        return;
      }

      // Live mode
      setLoadingDetails(true);
      setDetailError(null);
      try {
        const aggs = new Map<string, EvaluatorSummary>();
        let disagreements = 0;
        const cases: DisagreementCase[] = [];
        const agreements: EvaluatorAgreementSummary[] = [];

        for (const run of runs) {
          try {
            const sumBundle = await client.getRunSummaries(run.run_id);
            disagreements += sumBundle.evaluation_summary.disagreement_count;
            for (const [name, ev] of Object.entries(sumBundle.evaluation_summary.per_evaluator)) {
              const existing = aggs.get(name);
              if (!existing) {
                aggs.set(name, { ...ev });
              } else {
                existing.num_evaluated += ev.num_evaluated;
                existing.num_success += ev.num_success;
                existing.num_skipped += ev.num_skipped;
                existing.num_error += ev.num_error;
                existing.num_passed += ev.num_passed;
                existing.num_failed += ev.num_failed;
                existing.score_count += ev.score_count;
                if (ev.score_mean !== null) {
                  existing.score_mean =
                    existing.score_mean !== null
                      ? (existing.score_mean + ev.score_mean) / 2
                      : ev.score_mean;
                }
              }
            }
          } catch {
            // Optional
          }

          try {
            const evAnalytics = await client.getEvaluatorAnalytics(run.run_id);
            agreements.push(...evAnalytics.agreements);
          } catch {
            // Optional
          }

          if (run.result) {
            for (const tc of run.result.test_case_results) {
              const passedValues = tc.evaluation_results
                .map((e) => e.passed)
                .filter((p): p is boolean => p !== null);
              const uniquePassed = new Set(passedValues);
              if (uniquePassed.size > 1) {
                cases.push({
                  runId: run.run_id,
                  testCaseId: tc.test_case_id,
                  question: tc.input_text,
                  answer: tc.model_response?.output_text || tc.output_text || "",
                  results: tc.evaluation_results.map((e) => ({
                    name: e.evaluator_name,
                    passed: e.passed,
                    score: e.score,
                    label: e.label,
                  })),
                });
              }
            }
          }
        }

        setEvaluatorAggregates(aggs);
        setTotalDisagreements(disagreements);
        setDisagreementCases(cases);
        setAgreementsList(agreements);
      } catch (err: unknown) {
        setDetailError(err instanceof Error ? err : new Error(String(err)));
      } finally {
        setLoadingDetails(false);
      }
    }

    void loadData();
  }, [runs, dataMode, client]);

  const loading = runsLoading || loadingDetails;
  const error = runsError || detailError;

  return (
    <div>
      <SectionHeading eyebrow="Methodology comparison" title="Evaluator Comparison" />
      <FixtureNotice />

      <DataStateView loading={loading} error={error} onRetry={refetch}>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat label="Distinct Evaluators" value={evaluatorAggregates.size} />
          <Stat label="Total Disagreements" value={totalDisagreements} />
          <Stat label="Disagreement Cases" value={disagreementCases.length} />
          <Stat label="Evaluated Runs" value={runs?.length ?? 0} />
        </div>

        <div className="mt-8">
          <Card>
            <SectionHeading eyebrow="Aggregated metrics" title="Evaluator Coverage & Performance" />
            <p className="mb-4 text-xs text-ink-soft">
              Evaluator metrics across all recorded experiments. Evaluators assess distinct criteria
              (e.g. string equality, semantic proximity, claim support) and are never collapsed into a
              single omniscient score.
            </p>

            {evaluatorAggregates.size === 0 ? (
              <EmptyState title="No evaluator summary data recorded." />
            ) : (
              <div className="overflow-x-auto border border-line">
                <table className="w-full min-w-[760px] border-collapse text-sm">
                  <thead>
                    <tr className="border-b border-line bg-paper-dim text-left font-mono text-xs uppercase">
                      <th className="px-3 py-2">Evaluator</th>
                      <th className="px-3 py-2">Criterion</th>
                      <th className="px-3 py-2 text-right">Evaluated</th>
                      <th className="px-3 py-2 text-right">Success</th>
                      <th className="px-3 py-2 text-right">Skipped</th>
                      <th className="px-3 py-2 text-right">Errors</th>
                      <th className="px-3 py-2 text-right">Pass Rate</th>
                      <th className="px-3 py-2 text-right">Mean Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...evaluatorAggregates.values()].map((ev) => {
                      const passRate =
                        ev.num_success > 0 ? ev.num_passed / (ev.num_passed + ev.num_failed || 1) : null;
                      return (
                        <tr key={ev.evaluator_name} className="border-b border-line last:border-0">
                          <td className="px-3 py-2 font-mono text-xs font-medium">
                            {ev.evaluator_name}
                          </td>
                          <td className="px-3 py-2 font-mono text-xs text-ink-soft">
                            {ev.criterion ?? "unspecified"}
                          </td>
                          <td className="px-3 py-2 text-right font-mono text-xs">{ev.num_evaluated}</td>
                          <td className="px-3 py-2 text-right font-mono text-xs">{ev.num_success}</td>
                          <td className="px-3 py-2 text-right font-mono text-xs text-ink-soft">
                            {ev.num_skipped}
                          </td>
                          <td className="px-3 py-2 text-right font-mono text-xs">
                            {ev.num_error > 0 ? (
                              <span className="text-accent">{ev.num_error}</span>
                            ) : (
                              "0"
                            )}
                          </td>
                          <td className="px-3 py-2 text-right font-mono text-xs">
                            {formatPercent(passRate)}
                          </td>
                          <td className="px-3 py-2 text-right font-mono text-xs">
                            {formatScore(ev.score_mean)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </div>

        {/* Pairwise Evaluator Agreements */}
        {agreementsList.length > 0 && (
          <div className="mt-8">
            <Card>
              <SectionHeading eyebrow="Agreement analysis" title="Pairwise Evaluator Concordance" />
              <div className="overflow-x-auto border border-line">
                <table className="w-full min-w-[700px] border-collapse text-sm">
                  <thead>
                    <tr className="border-b border-line bg-paper-dim text-left font-mono text-xs uppercase">
                      <th className="px-3 py-2">Evaluator A</th>
                      <th className="px-3 py-2">Evaluator B</th>
                      <th className="px-3 py-2">Criterion</th>
                      <th className="px-3 py-2 text-right">Compared Cases</th>
                      <th className="px-3 py-2 text-right">Agreed</th>
                      <th className="px-3 py-2 text-right">Disagreed</th>
                      <th className="px-3 py-2 text-right">Concordance Rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {agreementsList.map((a, idx) => (
                      <tr key={idx} className="border-b border-line last:border-0">
                        <td className="px-3 py-2 font-mono text-xs">{a.evaluator_a}</td>
                        <td className="px-3 py-2 font-mono text-xs">{a.evaluator_b}</td>
                        <td className="px-3 py-2 font-mono text-xs text-ink-soft">{a.criterion}</td>
                        <td className="px-3 py-2 text-right font-mono text-xs">{a.cases_compared}</td>
                        <td className="px-3 py-2 text-right font-mono text-xs">{a.agreements}</td>
                        <td className="px-3 py-2 text-right font-mono text-xs">
                          {a.disagreements > 0 ? (
                            <span className="text-accent">{a.disagreements}</span>
                          ) : (
                            "0"
                          )}
                        </td>
                        <td className="px-3 py-2 text-right font-mono text-xs font-medium">
                          {formatPercent(a.agreement_rate)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>
        )}

        {/* Evaluator Disagreement Inspection Registry */}
        <div className="mt-8">
          <Card>
            <SectionHeading eyebrow="Disagreement analysis" title="Disagreement Investigation Registry" />
            <p className="mb-4 text-xs text-ink-soft">
              Cases where multiple evaluators evaluated the same output but produced conflicting
              verdicts (e.g. ExactMatch failed due to phrasing variation, while SemanticSimilarity
              passed).
            </p>

            {disagreementCases.length === 0 ? (
              <EmptyState title="No evaluator disagreements recorded across current runs." />
            ) : (
              <div className="space-y-4">
                {disagreementCases.map((c, idx) => (
                  <div key={idx} className="border border-line bg-paper-dim/40 p-4 text-xs">
                    <div className="mb-2 flex items-center justify-between font-mono">
                      <Link
                        to={`/runs/${c.runId}/test-cases/${c.testCaseId}`}
                        className="font-medium underline"
                      >
                        {c.testCaseId}
                      </Link>
                      <span className="text-ink-soft">Run: {shortFingerprint(c.runId, 12)}</span>
                    </div>

                    <div className="mb-2 text-ink">
                      <span className="font-semibold text-ink-soft">Question: </span>
                      {c.question}
                    </div>
                    <div className="mb-3 text-ink">
                      <span className="font-semibold text-ink-soft">Output: </span>
                      <code className="bg-paper-dim px-1 font-mono">{c.answer}</code>
                    </div>

                    <div className="flex flex-wrap gap-2">
                      {c.results.map((r) => (
                        <span
                          key={r.name}
                          className="flex items-center gap-1.5 border border-line bg-paper px-2 py-1 font-mono text-xs"
                        >
                          <span>{r.name}:</span>
                          <Tag tone={r.passed ? "neutral" : "negative"}>
                            {r.passed ? "pass" : "fail"}
                          </Tag>
                          {r.score !== null && (
                            <span className="text-ink-soft">({formatScore(r.score)})</span>
                          )}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </Card>
        </div>
      </DataStateView>
    </div>
  );
}
