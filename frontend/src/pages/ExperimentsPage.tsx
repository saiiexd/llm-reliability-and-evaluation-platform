import { useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { FixtureNotice } from "../components/FixtureNotice";
import { PageHeader } from "../components/Card";
import { EmptyState } from "../components/EmptyState";
import { DataStateView } from "../components/DataStateView";
import { NewExperimentForm } from "../components/NewExperimentForm";
import { PassBar } from "../components/Verdict";
import { useExperiments, useRuns } from "../data/useData";
import { experimentCaseCount, experimentModel, formatTimestamp, runPassCounts } from "../lib/derived";

export function ExperimentsPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [search, setSearch] = useState("");
  const [showForm, setShowForm] = useState(searchParams.get("new") === "true");
  const [createdId, setCreatedId] = useState<string | null>(null);

  const { data: experiments, loading: expLoading, error, refetch } = useExperiments(search);
  const { data: runs, loading: runsLoading } = useRuns();

  // Latest run per experiment, so each row can show how it last did.
  const latestRun = new Map<string, NonNullable<typeof runs>[number]>();
  for (const r of runs ?? []) {
    const prev = latestRun.get(r.experiment_id);
    if (!prev || r.started_at > prev.started_at) latestRun.set(r.experiment_id, r);
  }
  const expList = [...(experiments ?? [])].sort((a, b) => b.created_at.localeCompare(a.created_at));

  return (
    <div>
      <PageHeader
        title="Experiments"
        description="An experiment is a fixed set of questions, a model, and the checks used to judge its answers. Run one to get results."
        action={
          <button
            type="button"
            onClick={() => {
              setShowForm(!showForm);
              setCreatedId(null);
            }}
            className="rounded-md bg-ink px-4 py-2 text-sm font-medium text-paper hover:opacity-90"
          >
            {showForm ? "Close" : "New experiment"}
          </button>
        }
      />

      <FixtureNotice />

      {createdId && (
        <div className="mb-6 flex flex-col gap-3 rounded-xl border border-line bg-paper-dim p-4 sm:flex-row sm:items-center sm:justify-between">
          <p className="text-sm">
            Saved <span className="font-mono">{createdId}</span>. Next, run it to get results.
          </p>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => navigate(`/experiments/${createdId}`)}
              className="rounded-md bg-ink px-4 py-1.5 text-sm font-medium text-paper"
            >
              Open and run
            </button>
            <button type="button" onClick={() => setCreatedId(null)} className="px-2 text-sm text-ink-soft hover:text-ink">
              Dismiss
            </button>
          </div>
        </div>
      )}

      {showForm && (
        <div className="mb-8">
          <NewExperimentForm
            onCancel={() => setShowForm(false)}
            onCreated={(id) => {
              setCreatedId(id);
              setShowForm(false);
              void refetch();
            }}
          />
        </div>
      )}

      <input
        type="search"
        placeholder="Search experiments"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        className="mb-4 w-full max-w-sm rounded-md border border-line bg-paper px-3 py-1.5 text-sm placeholder:text-ink-soft focus:border-ink focus:outline-none"
      />

      <DataStateView loading={expLoading || runsLoading} error={error} onRetry={refetch}>
        {expList.length === 0 ? (
          <EmptyState
            title={search ? "No experiments match that search." : "No experiments yet."}
            detail={search ? "Try a different word." : "Create one with “New experiment”."}
          />
        ) : (
          <ul className="divide-y divide-line rounded-xl border border-line">
            {expList.map((exp) => {
              const last = latestRun.get(exp.experiment_id);
              const counts = last ? runPassCounts(last) : null;
              const model = experimentModel(exp);
              return (
                <li key={exp.experiment_id}>
                  <Link
                    to={`/experiments/${exp.experiment_id}`}
                    className="grid grid-cols-1 items-center gap-2 px-5 py-4 hover:bg-paper-dim sm:grid-cols-12 sm:gap-4"
                  >
                    <div className="min-w-0 sm:col-span-6">
                      <p className="truncate font-medium">{exp.name}</p>
                      <p className="truncate text-xs text-ink-soft">
                        {experimentCaseCount(exp)} questions
                        {model && (
                          <>
                            {" · "}
                            <span className="font-mono">{model}</span>
                          </>
                        )}
                        {" · created "}
                        {formatTimestamp(exp.created_at)}
                      </p>
                    </div>
                    <div className="sm:col-span-4">
                      {counts && counts.total > 0 ? (
                        <PassBar passed={counts.passed} total={counts.total} />
                      ) : (
                        <p className="text-xs text-ink-soft">Not run yet</p>
                      )}
                    </div>
                    <p className="text-sm tabular-nums text-ink-soft sm:col-span-2 sm:text-right">
                      {counts && counts.total > 0 ? (
                        <>
                          <span className="text-ink">{counts.passed}</span>/{counts.total} passed
                        </>
                      ) : (
                        "Open →"
                      )}
                    </p>
                  </Link>
                </li>
              );
            })}
          </ul>
        )}
      </DataStateView>
    </div>
  );
}
