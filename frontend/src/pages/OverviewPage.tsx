import { Link } from "react-router-dom";
import { FixtureNotice } from "../components/FixtureNotice";
import { Card, SectionHeading, Stat } from "../components/Card";
import { EmptyState } from "../components/EmptyState";
import { Tag } from "../components/Tag";
import { toneForFailureCategory, toneForRunStatus } from "../lib/tone";
import { formatTimestamp, isRagExecution, runCounts } from "../lib/derived";
import {
  getDiagnosticsForRun,
  getSummaryForRun,
  listAllRuns,
  listExperiments,
} from "../data/dataAccess";
import type { FailureCategory } from "../types/domain";

export function OverviewPage() {
  const experiments = listExperiments();
  const runs = listAllRuns();

  let numTestCases = 0;
  let numEvaluations = 0;
  let numRagRuns = 0;
  let numFailures = 0;
  let disagreementCount = 0;
  const evaluatorCoverage = new Map<string, number>();
  const categoryDistribution = new Map<FailureCategory, number>();

  for (const run of runs) {
    if (!run.result) continue;
    const counts = runCounts(run.result);
    numTestCases += counts.numTestCases;
    numFailures += counts.numFailed;
    if (run.result.test_case_results.some(isRagExecution)) numRagRuns += 1;

    for (const tcResult of run.result.test_case_results) {
      for (const evalResult of tcResult.evaluation_results) {
        numEvaluations += 1;
        evaluatorCoverage.set(
          evalResult.evaluator_name,
          (evaluatorCoverage.get(evalResult.evaluator_name) ?? 0) + 1,
        );
      }
    }

    const summary = getSummaryForRun(run.run_id);
    if (summary) disagreementCount += summary.evaluation_summary.disagreement_count;

    for (const diagnostic of Object.values(getDiagnosticsForRun(run.run_id))) {
      if (diagnostic.failure_category) {
        categoryDistribution.set(
          diagnostic.failure_category,
          (categoryDistribution.get(diagnostic.failure_category) ?? 0) + 1,
        );
      }
    }
  }

  const recentRuns = [...runs].sort((a, b) => (a.started_at < b.started_at ? 1 : -1)).slice(0, 5);

  return (
    <div>
      <SectionHeading eyebrow="Research console" title="Platform overview" />
      <FixtureNotice />

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Experiments" value={experiments.length} />
        <Stat label="Runs" value={runs.length} />
        <Stat label="Test cases" value={numTestCases} />
        <Stat label="Evaluations" value={numEvaluations} />
        <Stat label="RAG runs" value={numRagRuns} />
        <Stat label="Execution failures" value={numFailures} />
        <Stat label="Evaluator disagreements" value={disagreementCount} />
        <Stat label="Evaluation methods" value={evaluatorCoverage.size} />
      </div>

      <div className="mt-8 grid gap-6 lg:grid-cols-2">
        <Card>
          <SectionHeading eyebrow="Recent" title="Recent runs" />
          {recentRuns.length === 0 ? (
            <EmptyState title="No runs recorded yet." />
          ) : (
            <ul className="divide-y divide-line">
              {recentRuns.map((run) => (
                <li key={run.run_id} className="flex items-center justify-between py-2 text-sm">
                  <Link to={`/runs/${run.run_id}`} className="font-mono underline">
                    {run.run_id.slice(0, 12)}
                  </Link>
                  <div className="flex items-center gap-2">
                    <span className="text-ink-soft">{formatTimestamp(run.started_at)}</span>
                    <Tag tone={toneForRunStatus(run.status)}>{run.status}</Tag>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </Card>

        <Card>
          <SectionHeading eyebrow="Coverage" title="Evaluation method coverage" />
          {evaluatorCoverage.size === 0 ? (
            <EmptyState title="No evaluations recorded yet." />
          ) : (
            <ul className="divide-y divide-line">
              {[...evaluatorCoverage.entries()].map(([name, count]) => (
                <li key={name} className="flex items-center justify-between py-2 text-sm">
                  <span className="font-mono">{name}</span>
                  <span className="text-ink-soft">{count} evaluations</span>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>

      <div className="mt-6">
        <Card>
          <SectionHeading eyebrow="Diagnosis" title="Failure category distribution" />
          <p className="mb-3 text-xs text-ink-soft">
            Computed only from test cases where a reliability diagnosis exists; see the
            Reliability view for the full breakdown per run. Categories are never collapsed into
            a single reliability score.
          </p>
          {categoryDistribution.size === 0 ? (
            <EmptyState
              title="No consolidated reliability diagnoses available yet."
              detail="See the Reliability Diagnostics view for per-run detail."
            />
          ) : (
            <ul className="space-y-2">
              {[...categoryDistribution.entries()]
                .sort((a, b) => b[1] - a[1])
                .map(([category, count]) => {
                  const maxCount = Math.max(...categoryDistribution.values());
                  return (
                    <li key={category} className="flex items-center gap-3 text-sm">
                      <span className="w-44 shrink-0 font-mono text-xs">{category}</span>
                      <span className="h-3 flex-1 bg-paper-dim">
                        <span
                          className="block h-3 bg-ink"
                          style={{ width: `${(count / maxCount) * 100}%` }}
                        />
                      </span>
                      <span className="w-6 text-right text-ink-soft">{count}</span>
                    </li>
                  );
                })}
            </ul>
          )}
        </Card>
      </div>
    </div>
  );
}
