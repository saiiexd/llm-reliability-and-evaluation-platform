import { Link } from "react-router-dom";
import { FixtureNotice } from "../components/FixtureNotice";
import { PageHeader } from "../components/Card";
import { DataStateView } from "../components/DataStateView";
import { EmptyState } from "../components/EmptyState";
import { PassBar, VerdictBar, verdictSentence } from "../components/Verdict";
import { formatTimestamp, runPassCounts, verdictCounts } from "../lib/derived";
import { useExperiments, useRun, useRuns } from "../data/useData";
import type { ExperimentRun } from "../types/domain";

const STEPS = [
  {
    n: "1",
    title: "Define an experiment",
    body: "A set of questions with expected answers, the model to test, and the checks to run.",
  },
  {
    n: "2",
    title: "Run it",
    body: "The model answers every question. Each answer is checked automatically.",
  },
  {
    n: "3",
    title: "Read the results",
    body: "See which answers passed, which were wrong, and the likely reason.",
  },
];

function newestFirst(runs: ExperimentRun[]): ExperimentRun[] {
  return [...runs].sort((a, b) => b.started_at.localeCompare(a.started_at));
}

export function OverviewPage() {
  const experiments = useExperiments();
  const runs = useRuns();

  const loading = experiments.loading || runs.loading;
  const error = experiments.error || runs.error;
  const retry = () => {
    void experiments.refetch();
    void runs.refetch();
  };

  const expList = experiments.data ?? [];
  const runList = newestFirst((runs.data ?? []).filter((r) => runPassCounts(r).total > 0));
  const nameOf = new Map(expList.map((e) => [e.experiment_id, e.name]));

  const overall = runList.reduce(
    (acc, r) => {
      const c = runPassCounts(r);
      return { passed: acc.passed + c.passed, total: acc.total + c.total };
    },
    { passed: 0, total: 0 },
  );

  // List endpoints only return summaries; fetch the latest run in full so
  // its result can be split into passed / wrong / not scored / crashed.
  const latestId = runList[0]?.run_id ?? "";
  const { data: latest } = useRun(latestId);
  const latestCounts = latest?.result ? verdictCounts(latest.result.test_case_results) : null;

  return (
    <div>
      <PageHeader
        title="Is the model answering correctly?"
        description="This platform asks a language model a fixed set of questions, checks every answer automatically, and explains why the wrong ones went wrong."
        action={
          <>
            <Link
              to="/experiments?new=true"
              className="rounded-md bg-ink px-4 py-2 text-sm font-medium text-paper hover:opacity-90"
            >
              Start an Experiment
            </Link>
            <Link
              to="/results"
              className="rounded-md border border-line px-4 py-2 text-sm text-ink hover:bg-paper-dim"
            >
              View results
            </Link>
          </>
        }
      />

      <ol className="mb-10 grid gap-3 sm:grid-cols-3">
        {STEPS.map((s) => (
          <li key={s.n} className="rounded-xl border border-line p-4">
            <p className="text-xs text-ink-soft">Step {s.n}</p>
            <p className="mt-1 font-medium">{s.title}</p>
            <p className="mt-1 text-sm leading-relaxed text-ink-soft">{s.body}</p>
          </li>
        ))}
      </ol>

      <FixtureNotice />

      <DataStateView loading={loading} error={error} onRetry={retry}>
        {runList.length === 0 ? (
          <EmptyState
            title="No results yet"
            detail="Create an experiment and run it. Its results will show up here."
          />
        ) : (
          <div className="space-y-10">
            <section className="grid gap-6 lg:grid-cols-2">
              <div className="rounded-xl border border-line bg-paper-dim p-6">
                <p className="text-sm text-ink-soft">All runs so far</p>
                <p className="mt-1 text-4xl font-semibold tabular-nums">
                  {overall.total > 0 ? Math.round((overall.passed / overall.total) * 100) : 0}%
                  <span className="ml-2 text-base font-normal text-ink-soft">passed</span>
                </p>
                <p className="mt-2 text-sm text-ink-soft">
                  {overall.total} answers checked across {runList.length} runs of{" "}
                  {expList.length} experiments.
                </p>
                <div className="mt-5">
                  <PassBar passed={overall.passed} total={overall.total} />
                </div>
                <p className="mt-3 text-xs leading-relaxed text-ink-soft">
                  "Not passed" includes wrong answers and answers a check couldn't score. Open a run
                  to see which is which.
                </p>
              </div>

              {latest && latestCounts && (
                <div className="flex flex-col rounded-xl border border-line p-6">
                  <p className="text-sm text-ink-soft">Latest run · {formatTimestamp(latest.started_at)}</p>
                  <p className="mt-1 text-lg font-semibold">
                    {nameOf.get(latest.experiment_id) ?? latest.experiment_id}
                  </p>
                  <p className="text-sm text-ink-soft">
                    Model <span className="font-mono text-ink">{latest.model_config?.model_id}</span>
                  </p>
                  <p className="mt-4 text-sm">{verdictSentence(latestCounts)}</p>
                  <div className="mt-3">
                    <VerdictBar counts={latestCounts} showLegend={false} />
                  </div>
                  <Link
                    to={`/results/${latest.run_id}`}
                    className="mt-auto self-start pt-5 text-sm font-medium underline-offset-4 hover:underline"
                  >
                    Open this run →
                  </Link>
                </div>
              )}
            </section>

            <section>
              <div className="mb-3 flex items-baseline justify-between">
                <h2 className="text-lg font-semibold">Recent runs</h2>
                <Link to="/results" className="text-sm text-ink-soft hover:text-ink">
                  All results →
                </Link>
              </div>
              <ul className="divide-y divide-line rounded-xl border border-line">
                {runList.slice(0, 6).map((run) => {
                  const counts = runPassCounts(run);
                  return (
                    <li key={run.run_id}>
                      <Link
                        to={`/results/${run.run_id}`}
                        className="grid grid-cols-1 items-center gap-2 px-5 py-4 hover:bg-paper-dim sm:grid-cols-12 sm:gap-4"
                      >
                        <div className="min-w-0 sm:col-span-5">
                          <p className="truncate font-medium">
                            {nameOf.get(run.experiment_id) ?? run.experiment_id}
                          </p>
                          <p className="text-xs text-ink-soft">
                            {formatTimestamp(run.started_at)} · <span className="font-mono">{run.run_id.slice(0, 8)}</span>
                          </p>
                        </div>
                        <div className="sm:col-span-5">
                          <PassBar passed={counts.passed} total={counts.total} />
                        </div>
                        <p className="text-sm tabular-nums text-ink-soft sm:col-span-2 sm:text-right">
                          <span className="text-ink">{counts.passed}</span>/{counts.total} passed
                        </p>
                      </Link>
                    </li>
                  );
                })}
              </ul>
            </section>
          </div>
        )}
      </DataStateView>
    </div>
  );
}
