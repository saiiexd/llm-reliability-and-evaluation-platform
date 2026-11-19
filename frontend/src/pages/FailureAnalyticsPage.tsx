import { useState } from "react";
import { FixtureNotice } from "../components/FixtureNotice";
import { Card, SectionHeading } from "../components/Card";
import { Tag } from "../components/Tag";
import { useFailureAnalytics, useRuns } from "../data/useData";
import { toneForFailureCategory } from "../lib/tone";
import { DataStateView } from "../components/DataStateView";
import type { FailureCategory } from "../types/domain";

export function FailureAnalyticsPage() {
  const { data: runs, loading: runsLoading } = useRuns();
  const [selectedRunId, setSelectedRunId] = useState<string>("");

  const effectiveRunId = selectedRunId || (runs && runs[0]?.run_id) || "";
  const { data: failureDist, loading: distLoading, error, refetch } = useFailureAnalytics(effectiveRunId);

  const runList = runs || [];

  return (
    <div>
      <SectionHeading
        eyebrow="Taxonomy analytics"
        title="Failure Forensics & Diagnostic Distribution"
        description="Independent aggregation of primary failure categories and co-occurring secondary findings, preserving underlying counts, denominators, and multi-signal evidence."
      />
      <FixtureNotice />

      {/* Run Selector */}
      <div className="mb-6 flex items-center gap-3">
        <label htmlFor="run-selector" className="font-mono text-xs text-ink-soft">
          Select Experiment Run:
        </label>
        <select
          id="run-selector"
          value={effectiveRunId}
          onChange={(e) => setSelectedRunId(e.target.value)}
          className="border border-line bg-paper px-3 py-1.5 font-mono text-xs text-ink"
        >
          {runList.map((r) => (
            <option key={r.run_id} value={r.run_id}>
              {r.run_id.slice(0, 12)} ({r.experiment_id})
            </option>
          ))}
        </select>
      </div>

      <DataStateView loading={runsLoading || distLoading} error={error} onRetry={refetch}>
        {!failureDist ? (
          <Card>
            <p className="text-sm text-ink-soft">No diagnostic records found for this run.</p>
          </Card>
        ) : (
          <div className="space-y-6">
            {/* Summary Rollup */}
            <div className="grid grid-cols-3 gap-3">
              <Card>
                <div className="text-xs text-ink-soft">Total diagnosed cases</div>
                <div className="mt-1 font-mono text-xl font-medium">{failureDist.total_cases}</div>
              </Card>
              <Card>
                <div className="text-xs text-ink-soft">Determined outcomes</div>
                <div className="mt-1 font-mono text-xl font-medium">
                  {failureDist.num_determined} / {failureDist.total_cases}
                </div>
              </Card>
              <Card>
                <div className="text-xs text-ink-soft">Undetermined / evidence gaps</div>
                <div className="mt-1 font-mono text-xl font-medium">
                  {failureDist.num_undetermined} / {failureDist.total_cases}
                </div>
              </Card>
            </div>

            {/* Primary Category Distribution */}
            <Card>
              <SectionHeading
                eyebrow="Primary assignment"
                title="Primary Failure Category Distribution"
                description="Mutually exclusive primary categorization synthesized according to strict scientific precedence (execution error > retrieval failure > unfaithful generation > incorrect answer > correct)."
              />
              <div className="space-y-3">
                {Object.entries(failureDist.primary_categories).map(([cat, stats]) => {
                  const propVal = stats.proportion ?? 0;
                  return (
                    <div key={cat} className="border-b border-line pb-2 last:border-0">
                      <div className="flex items-center justify-between text-sm">
                        <Tag tone={toneForFailureCategory(cat as FailureCategory)}>{cat}</Tag>
                        <span className="font-mono text-xs text-ink">
                          {stats.count} cases ({(propVal * 100).toFixed(1)}%)
                        </span>
                      </div>
                      <div className="mt-1.5 h-1.5 w-full bg-paper-dim">
                        <div
                          className="h-1.5 bg-ink"
                          style={{ width: `${propVal * 100}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </Card>

            {/* Co-occurring Secondary Findings */}
            <Card>
              <SectionHeading
                eyebrow="Multi-signal evidence"
                title="Secondary Observations & Co-occurrence"
                description="Secondary observations preserved alongside the primary diagnosis. A single primary assignment does not discard secondary findings such as evaluator disagreements or unsupported claims."
              />
              {Object.keys(failureDist.secondary_findings).length === 0 ? (
                <p className="text-xs text-ink-soft">No secondary findings recorded for this run.</p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b border-line text-ink-soft font-mono uppercase">
                        <th className="py-2 pr-4 font-normal">Secondary Observation</th>
                        <th className="py-2 pr-4 font-normal text-right font-mono">Occurrences</th>
                        <th className="py-2 pr-4 font-normal text-right font-mono">Proportion</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-line font-mono">
                      {Object.entries(failureDist.secondary_findings).map(([finding, stats]) => (
                        <tr key={finding}>
                          <td className="py-2 pr-4 font-sans text-xs">{finding}</td>
                          <td className="py-2 pr-4 text-right">{stats.count}</td>
                          <td className="py-2 pr-4 text-right">
                            {stats.frequency !== null ? `${(stats.frequency * 100).toFixed(1)}%` : "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Card>
          </div>
        )}
      </DataStateView>
    </div>
  );
}
