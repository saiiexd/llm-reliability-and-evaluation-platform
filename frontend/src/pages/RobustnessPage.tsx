import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { FixtureNotice } from "../components/FixtureNotice";
import { Card, SectionHeading, Stat } from "../components/Card";
import { EmptyState } from "../components/EmptyState";
import { Tag } from "../components/Tag";
import { toneForFailureCategory } from "../lib/tone";
import { formatPercent } from "../lib/derived";
import { useConnection } from "../context/ConnectionContext";
import { useRuns } from "../data/useData";
import * as fixtures from "../data/dataAccess";
import { DataStateView } from "../components/DataStateView";
import type { FailureCategory, PerturbationType, RobustnessResult } from "../types/domain";

export function RobustnessPage() {
  const { dataMode, client } = useConnection();
  const { data: runs, loading: runsLoading, error: runsError, refetch } = useRuns();

  interface ComparisonRow extends RobustnessResult {
    runId: string;
  }

  const [allComparisons, setAllComparisons] = useState<ComparisonRow[]>([]);
  const [transitionCounts, setTransitionCounts] = useState<Record<string, number>>({});
  const [perturbationStats, setPerturbationStats] = useState<
    Map<PerturbationType, { total: number; unchanged: number }>
  >(new Map());
  const [totalComparisons, setTotalComparisons] = useState(0);
  const [totalUnchanged, setTotalUnchanged] = useState(0);
  const [loadingDetails, setLoadingDetails] = useState(false);
  const [detailError, setDetailError] = useState<Error | null>(null);

  useEffect(() => {
    async function loadData() {
      if (!runs) return;

      if (dataMode === "fixture") {
        const comps: ComparisonRow[] = [];
        const trans: Record<string, number> = {};
        const stats = new Map<PerturbationType, { total: number; unchanged: number }>();
        let numComp = 0;
        let numUnchanged = 0;

        for (const run of runs) {
          const bundle = fixtures.getRobustnessForRun(run.run_id);
          if (!bundle || !bundle.consistency) continue;

          const c = bundle.consistency as unknown as Record<string, unknown>;
          const cTotal = Number(c.total_comparisons ?? c.num_comparisons ?? 0) || 0;
          const cUnchanged = Number(c.identical_outcomes ?? c.num_unchanged ?? 0) || 0;

          numComp += cTotal;
          numUnchanged += cUnchanged;

          const rawTrans = c.category_transitions || c.transition_counts || {};
          for (const [t, count] of Object.entries(rawTrans)) {
            trans[t] = (trans[t] ?? 0) + (Number(count) || 0);
          }

          for (const comp of bundle.comparisons) {
            comps.push({ ...comp, runId: run.run_id });
            const current = stats.get(comp.perturbation_type) ?? { total: 0, unchanged: 0 };
            current.total += 1;
            if (!comp.outcome_changed) current.unchanged += 1;
            stats.set(comp.perturbation_type, current);
          }
        }

        setAllComparisons(comps);
        setTransitionCounts(trans);
        setPerturbationStats(stats);
        setTotalComparisons(numComp);
        setTotalUnchanged(numUnchanged);
        return;
      }

      // Live mode
      setLoadingDetails(true);
      setDetailError(null);
      try {
        const comps: ComparisonRow[] = [];
        const trans: Record<string, number> = {};
        const stats = new Map<PerturbationType, { total: number; unchanged: number }>();
        let numComp = 0;
        let numUnchanged = 0;

        for (const run of runs) {
          try {
            const bundle = await client.getRunRobustness(run.run_id);
            if (!bundle || !bundle.consistency) continue;

            const c = bundle.consistency as unknown as Record<string, unknown>;
            const cTotal = Number(c.total_comparisons ?? c.num_comparisons ?? 0) || 0;
            const cUnchanged = Number(c.identical_outcomes ?? c.num_unchanged ?? 0) || 0;

            numComp += cTotal;
            numUnchanged += cUnchanged;

            const rawTrans = c.category_transitions || c.transition_counts || {};
            for (const [t, count] of Object.entries(rawTrans)) {
              trans[t] = (trans[t] ?? 0) + (Number(count) || 0);
            }

            for (const comp of bundle.comparisons) {
              comps.push({
                runId: run.run_id,
                baseline_test_case_id: comp.baseline_test_case_id,
                perturbed_test_case_id: comp.perturbed_test_case_id,
                perturbation_type: comp.perturbation_type as PerturbationType,
                baseline_category: comp.baseline_category as FailureCategory | null,
                perturbed_category: comp.perturbed_category as FailureCategory | null,
                outcome_changed: comp.outcome_changed,
                failure_category_transition: comp.failure_category_transition,
              });

              const pType = comp.perturbation_type as PerturbationType;
              const current = stats.get(pType) ?? { total: 0, unchanged: 0 };
              current.total += 1;
              if (!comp.outcome_changed) current.unchanged += 1;
              stats.set(pType, current);
            }
          } catch {
            // Optional if run has no robustness
          }
        }

        setAllComparisons(comps);
        setTransitionCounts(trans);
        setPerturbationStats(stats);
        setTotalComparisons(numComp);
        setTotalUnchanged(numUnchanged);
      } catch (err: unknown) {
        setDetailError(err instanceof Error ? err : new Error(String(err)));
      } finally {
        setLoadingDetails(false);
      }
    }

    void loadData();
  }, [runs, dataMode, client]);

  const consistencyRate = totalComparisons > 0 ? totalUnchanged / totalComparisons : null;

  const loading = runsLoading || loadingDetails;
  const error = runsError || detailError;

  return (
    <div>
      <SectionHeading title="Does rewording change the verdict?" description="Each pair is the same question asked two ways. A consistent model gets the same diagnosis for both." />
      <FixtureNotice />

      <DataStateView loading={loading} error={error} onRetry={refetch}>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat label="Question pairs tested" value={totalComparisons} />
          <Stat label="Same verdict" value={totalUnchanged} />
          <Stat label="Verdict changed" value={totalComparisons - totalUnchanged} />
          <Stat label="Consistency" value={formatPercent(consistencyRate)} />
        </div>

        <div className="mt-8 grid gap-6 lg:grid-cols-2">
          <Card>
            <SectionHeading title="By type of change" />
            {perturbationStats.size === 0 ? (
              <EmptyState title="No perturbation comparisons recorded." />
            ) : (
              <div className="space-y-3">
                {[...perturbationStats.entries()].map(([type, stats]) => {
                  const rate = stats.total > 0 ? stats.unchanged / stats.total : 0;
                  return (
                    <div key={type} className="border-b border-line pb-2 last:border-0">
                      <div className="flex items-center justify-between text-sm">
                        <span className="font-mono text-xs">{type}</span>
                        <span className="font-mono text-xs font-medium">{formatPercent(rate)}</span>
                      </div>
                      <div className="mt-1 flex items-center justify-between text-[11px] text-ink-soft">
                        <span>
                          {stats.unchanged} / {stats.total} unchanged
                        </span>
                        <span>{stats.total - stats.unchanged} outcome shifts</span>
                      </div>
                      <div className="mt-1.5 h-1.5 w-full bg-paper-dim">
                        <div className="h-1.5 bg-ink" style={{ width: `${rate * 100}%` }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </Card>

          <Card>
            <SectionHeading title="How verdicts changed" description="Pairs where the change flipped the diagnosis, e.g. correct → wrong." />
            <p className="mb-3 text-xs text-ink-soft">
              Transitions where a perturbation caused the model output to shift from one failure
              category to another.
            </p>
            {Object.keys(transitionCounts).length === 0 ? (
              <EmptyState title="No verdict changed." />
            ) : (
              <ul className="space-y-2">
                {Object.entries(transitionCounts).map(([transition, count]) => (
                  <li key={transition} className="flex items-center justify-between text-xs">
                    <span className="font-mono">{transition}</span>
                    <span className="font-mono text-ink-soft">{count} times</span>
                  </li>
                ))}
              </ul>
            )}
          </Card>
        </div>

        <div className="mt-8">
          <Card>
            <SectionHeading title="Every pair" />
            {allComparisons.length === 0 ? (
              <EmptyState title="No perturbation test cases recorded." />
            ) : (
              <div className="overflow-x-auto border border-line">
                <table className="w-full min-w-[760px] border-collapse text-sm">
                  <thead>
                    <tr className="border-b border-line bg-paper-dim text-left text-xs text-ink-soft font-normal">
                      <th className="px-3 py-2">Original question</th>
                      <th className="px-3 py-2">Changed question</th>
                      <th className="px-3 py-2">Change</th>
                      <th className="px-3 py-2">Original verdict</th>
                      <th className="px-3 py-2">New verdict</th>
                      <th className="px-3 py-2">Result</th>
                    </tr>
                  </thead>
                  <tbody>
                    {allComparisons.map((c, idx) => (
                      <tr
                        key={idx}
                        className="border-b border-line hover:bg-paper-dim/40 last:border-0"
                      >
                        <td className="px-3 py-2 font-mono text-xs">
                          <Link
                            to={`/runs/${c.runId}/test-cases/${c.baseline_test_case_id}`}
                            className="underline"
                          >
                            {c.baseline_test_case_id}
                          </Link>
                        </td>
                        <td className="px-3 py-2 font-mono text-xs">
                          <Link
                            to={`/runs/${c.runId}/test-cases/${c.perturbed_test_case_id}`}
                            className="underline"
                          >
                            {c.perturbed_test_case_id}
                          </Link>
                        </td>
                        <td className="px-3 py-2 font-mono text-xs text-ink-soft">
                          {c.perturbation_type}
                        </td>
                        <td className="px-3 py-2">
                          {c.baseline_category ? (
                            <Tag tone={toneForFailureCategory(c.baseline_category as FailureCategory)}>
                              {c.baseline_category}
                            </Tag>
                          ) : (
                            <span className="font-mono text-xs text-ink-soft">none</span>
                          )}
                        </td>
                        <td className="px-3 py-2">
                          {c.perturbed_category ? (
                            <Tag tone={toneForFailureCategory(c.perturbed_category as FailureCategory)}>
                              {c.perturbed_category}
                            </Tag>
                          ) : (
                            <span className="font-mono text-xs text-ink-soft">none</span>
                          )}
                        </td>
                        <td className="px-3 py-2">
                          <Tag tone={c.outcome_changed ? "caution" : "neutral"}>
                            {c.outcome_changed ? "shifted" : "identical"}
                          </Tag>
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
