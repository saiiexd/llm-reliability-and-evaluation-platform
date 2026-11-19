import { useEffect, useState, type ReactNode } from "react";
import { Link, useParams } from "react-router-dom";
import { SimulatedTag } from "../components/SimulatedNotice";
import { FixtureNotice } from "../components/FixtureNotice";
import { PageHeader } from "../components/Card";
import { EmptyState } from "../components/EmptyState";
import { DataStateView } from "../components/DataStateView";
import { PassBar } from "../components/Verdict";
import {
  evaluatorLabel,
  experimentCaseCount,
  formatTimestamp,
  runPassCounts,
  shortFingerprint,
} from "../lib/derived";
import { useExperiment, useRuns } from "../data/useData";
import { useConnection } from "../context/ConnectionContext";
import type { RunExecutionResponse } from "../api/client";
import type { BenchmarkExecutionResponse, ExperimentDefinition, ProviderStatus } from "../types/domain";

const ADAPTERS = [
  { id: "rag-pipeline-v1", label: "RAG pipeline (top 1 passage)", isExternal: false },
  { id: "rag-pipeline-topk-2", label: "RAG pipeline (top 2 passages)", isExternal: false },
  { id: "mock-adapter-v1", label: "Mock model (offline)", isExternal: false },
  { id: "echo-adapter", label: "Echo (repeats the question)", isExternal: false },
  { id: "openai-gpt-4o-mini", label: "OpenAI GPT-4o mini", isExternal: true },
  { id: "openai-gpt-4o", label: "OpenAI GPT-4o", isExternal: true },
  { id: "openai-custom", label: "OpenAI-compatible server (vLLM, Ollama…)", isExternal: true },
];

const isExternal = (id: string) => ADAPTERS.find((a) => a.id === id)?.isExternal ?? false;

function Section({ title, hint, children }: { title: string; hint?: string; children: ReactNode }) {
  return (
    <section className="rounded-xl border border-line p-6">
      <h2 className="font-semibold">{title}</h2>
      {hint && <p className="mt-0.5 text-sm text-ink-soft">{hint}</p>}
      <div className="mt-4">{children}</div>
    </section>
  );
}

function ExternalConsent({
  checked,
  onChange,
  disabled,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  disabled: boolean;
}) {
  return (
    <label className="flex items-start gap-2 rounded-lg border border-line p-3 text-sm">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        disabled={disabled}
        className="mt-1 accent-ink"
      />
      <span className="text-ink-soft">
        <span className="text-ink">This calls a paid external API.</span> The server needs{" "}
        <code className="font-mono text-ink">OPENAI_API_KEY</code> set. Tick to allow it.
      </span>
    </label>
  );
}

function RunControls({ experimentId, onRan }: { experimentId: string; onRan: () => void }) {
  const { dataMode, executeRun, executeBenchmark, client } = useConnection();
  const [mode, setMode] = useState<"single" | "benchmark">("single");
  const [adapter, setAdapter] = useState("rag-pipeline-v1");
  const [benchAdapters, setBenchAdapters] = useState<string[]>(["mock-adapter-v1", "echo-adapter"]);
  const [repetitions, setRepetitions] = useState(1);
  const [allowExternal, setAllowExternal] = useState(false);
  const [running, setRunning] = useState(false);
  const [single, setSingle] = useState<RunExecutionResponse | null>(null);
  const [bench, setBench] = useState<BenchmarkExecutionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [providers, setProviders] = useState<ProviderStatus[]>([]);

  useEffect(() => {
    if (dataMode === "live") {
      client
        .listProviders()
        .then((res) => setProviders(res.providers))
        .catch(() => {
          // Provider status is optional; the run itself will report a missing key.
        });
    }
  }, [client, dataMode]);

  const needsExternal = mode === "single" ? isExternal(adapter) : benchAdapters.some(isExternal);
  const openAiReady = providers.find((p) => p.provider_id === "openai")?.is_configured ?? false;
  const live = dataMode === "live";

  const run = async () => {
    setRunning(true);
    setSingle(null);
    setBench(null);
    setError(null);
    try {
      if (mode === "single") {
        setSingle(await executeRun(experimentId, adapter, allowExternal));
      } else {
        setBench(
          await executeBenchmark({
            experiment_id: experimentId,
            adapter_ids: benchAdapters,
            repetitions,
            allow_external_inference: allowExternal,
          }),
        );
      }
      onRan();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
    }
  };

  const disabled =
    running || !live || (mode === "benchmark" && benchAdapters.length === 0) || (needsExternal && !allowExternal);

  return (
    <Section title="Run it" hint="Ask the model every question in this experiment and check the answers.">
      {!live && (
        <p className="mb-4 rounded-lg border border-dashed border-ink-soft/60 p-3 text-sm text-ink-soft">
          Running needs the live backend. You're viewing demo data.
        </p>
      )}

      <div className="mb-4 flex gap-1">
        {(["single", "benchmark"] as const).map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => setMode(m)}
            className={`rounded-md px-3 py-1.5 text-sm ${
              mode === m ? "bg-paper-raised font-medium" : "text-ink-soft hover:text-ink"
            }`}
          >
            {m === "single" ? "One model" : "Compare models"}
          </button>
        ))}
      </div>

      {mode === "single" ? (
        <label className="block max-w-md text-sm">
          <span className="mb-1 block text-ink-soft">Model</span>
          <select
            id="adapter-select"
            value={adapter}
            onChange={(e) => setAdapter(e.target.value)}
            disabled={running || !live}
            className="w-full rounded-md border border-line bg-paper px-3 py-2 text-sm disabled:opacity-50"
          >
            {ADAPTERS.map((a) => (
              <option key={a.id} value={a.id}>
                {a.label}
              </option>
            ))}
          </select>
        </label>
      ) : (
        <div className="space-y-4">
          <div className="grid gap-2 sm:grid-cols-2">
            {ADAPTERS.map((a) => {
              const on = benchAdapters.includes(a.id);
              return (
                <label
                  key={a.id}
                  className={`flex cursor-pointer items-center gap-2 rounded-lg border px-3 py-2 text-sm ${
                    on ? "border-ink bg-paper-raised" : "border-line"
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={on}
                    onChange={() =>
                      setBenchAdapters((prev) => (on ? prev.filter((x) => x !== a.id) : [...prev, a.id]))
                    }
                    disabled={running}
                    className="accent-ink"
                  />
                  {a.label}
                </label>
              );
            })}
          </div>
          <label className="flex flex-wrap items-center gap-3 text-sm">
            <span className="text-ink-soft">Repeat each model</span>
            <input
              id="repetition-input"
              type="number"
              min={1}
              max={5}
              value={repetitions}
              onChange={(e) => setRepetitions(Math.max(1, Math.min(5, parseInt(e.target.value) || 1)))}
              disabled={running || !live}
              className="w-16 rounded-md border border-line bg-paper px-2 py-1"
            />
            <span className="text-xs text-ink-soft">times. Repeats show whether answers change between runs.</span>
          </label>
        </div>
      )}

      {needsExternal && (
        <div className="mt-4 space-y-2">
          {live && !openAiReady && (
            <p className="text-sm text-accent">The server has no OpenAI key configured, so this will fail.</p>
          )}
          <ExternalConsent checked={allowExternal} onChange={setAllowExternal} disabled={running} />
        </div>
      )}

      <button
        type="button"
        onClick={run}
        disabled={disabled}
        className="mt-5 rounded-md bg-ink px-5 py-2 text-sm font-medium text-paper hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {running
          ? "Running…"
          : mode === "single"
            ? "Run"
            : `Run ${benchAdapters.length * repetitions} runs`}
      </button>

      {single && (
        <p className="mt-4 text-sm">
          Done in {single.duration_seconds.toFixed(1)}s.{" "}
          <Link to={`/results/${single.run_id}`} className="font-medium underline underline-offset-4">
            See the results →
          </Link>
        </p>
      )}
      {bench && (
        <div className="mt-4 text-sm">
          <p>Finished {bench.total_runs} runs.</p>
          <ul className="mt-2 space-y-1">
            {bench.runs.map((r) => (
              <li key={r.run_id}>
                <Link to={`/results/${r.run_id}`} className="font-mono text-xs underline underline-offset-4">
                  {r.run_id}
                </Link>
              </li>
            ))}
          </ul>
        </div>
      )}
      {error && <p className="mt-4 text-sm text-accent">Run failed: {error}</p>}
    </Section>
  );
}

function SetupDetails({ experiment }: { experiment: ExperimentDefinition }) {
  const mc = experiment.model_config;
  const rag = (mc?.extra_params?.rag ?? null) as Record<string, unknown> | null;
  const rows: [string, ReactNode][] = [
    ["Experiment ID", <span className="font-mono">{experiment.experiment_id}</span>],
    ["Created", formatTimestamp(experiment.created_at)],
    ["Config fingerprint", <span className="font-mono">{shortFingerprint(experiment.config_fingerprint)}</span>],
    ["Dataset", experiment.dataset_version?.dataset_name ?? "n/a"],
    ["Dataset version", <span className="font-mono">{shortFingerprint(experiment.dataset_version?.version_id ?? "")}</span>],
    ["Model", <span className="font-mono">{mc?.model_id ?? "n/a"}</span>],
    ["Temperature", mc?.temperature ?? "default"],
    ["Max tokens", mc?.max_tokens ?? "default"],
  ];
  if (rag) {
    rows.push(
      ["Embedding model", <span className="font-mono">{String(rag.embedding_model_id ?? "")}</span>],
      ["Chunk size / overlap", `${String(rag.chunk_size ?? "")} / ${String(rag.chunk_overlap ?? "")}`],
      ["Passages retrieved (top K)", String(rag.top_k ?? "")],
    );
  }
  return (
    <details className="group rounded-xl border border-line">
      <summary className="flex cursor-pointer select-none items-center gap-2 px-6 py-4 text-sm hover:bg-paper-dim">
        <span className="text-ink-soft transition-transform group-open:rotate-90">›</span>
        Technical setup
        <span className="text-ink-soft">(IDs, model settings, check parameters)</span>
      </summary>
      <div className="border-t border-line px-6 py-4 text-sm">
        <dl className="divide-y divide-line">
          {rows.map(([k, v]) => (
            <div key={k} className="flex justify-between gap-4 py-2">
              <dt className="text-ink-soft">{k}</dt>
              <dd className="min-w-0 text-right break-words">{v}</dd>
            </div>
          ))}
        </dl>
        {experiment.evaluator_configs?.some((e) => Object.keys(e.parameters ?? {}).length > 0) && (
          <div className="mt-4 space-y-3">
            {experiment.evaluator_configs.map((e) =>
              Object.keys(e.parameters ?? {}).length === 0 ? null : (
                <div key={e.evaluator_name}>
                  <p className="text-ink-soft">{evaluatorLabel(e.evaluator_name)} parameters</p>
                  <pre className="mt-1 max-h-64 overflow-y-auto rounded-lg bg-paper-dim p-3 font-mono text-xs text-ink-soft">
                    {JSON.stringify(e.parameters, null, 2)}
                  </pre>
                </div>
              ),
            )}
          </div>
        )}
      </div>
    </details>
  );
}

export function ExperimentDetailPage() {
  const { experimentId = "" } = useParams<{ experimentId: string }>();
  const { data: experiment, loading: expLoading, error: expError, refetch: refetchExp } = useExperiment(experimentId);
  const { data: runs, loading: runsLoading, error: runsError, refetch: refetchRuns } = useRuns(experimentId);
  const [showAllQuestions, setShowAllQuestions] = useState(false);

  const runList = [...(runs ?? [])].sort((a, b) => b.started_at.localeCompare(a.started_at));
  const questions = experiment?.dataset_version?.test_cases ?? [];
  const shownQuestions = showAllQuestions ? questions : questions.slice(0, 5);

  return (
    <div>
      <Link to="/experiments" className="mb-4 inline-block text-sm text-ink-soft hover:text-ink">
        ← Experiments
      </Link>
      <DataStateView
        loading={expLoading || runsLoading}
        error={expError || runsError}
        onRetry={() => {
          void refetchExp();
          void refetchRuns();
        }}
      >
        {!experiment ? (
          <EmptyState title="Experiment not found" detail={`There is no experiment with ID "${experimentId}".`} />
        ) : (
          <>
            <PageHeader
              title={experiment.name}
              description={
                experiment.description ??
                `${experimentCaseCount(experiment)} questions, answered by ${experiment.model_config?.model_id ?? "a model"} and judged by ${
                  experiment.evaluator_configs?.map((e) => evaluatorLabel(e.evaluator_name).toLowerCase()).join(" and ") ||
                  "no checks"
                }.`
              }
            />
            <FixtureNotice />

            <div className="space-y-6">
              <RunControls experimentId={experimentId} onRan={() => void refetchRuns()} />

              <Section title={`Past runs (${runList.length})`}>
                {runList.length === 0 ? (
                  <p className="text-sm text-ink-soft">Not run yet.</p>
                ) : (
                  <ul className="-mx-6 divide-y divide-line border-y border-line">
                    {runList.map((run) => {
                      const c = runPassCounts(run);
                      return (
                        <li key={run.run_id}>
                          <Link
                            to={`/results/${run.run_id}`}
                            className="grid grid-cols-1 items-center gap-2 px-6 py-3 hover:bg-paper-dim sm:grid-cols-12 sm:gap-4"
                          >
                            <p className="text-sm sm:col-span-4">
                              {formatTimestamp(run.started_at)}{" "}
                              <span className="font-mono text-xs text-ink-soft">{run.run_id.slice(0, 8)}</span>
                              {run.adapter_id && (
                                <span className="font-mono text-xs text-ink-soft"> · {run.adapter_id}</span>
                              )}
                              {run.benchmark_id && (
                                <span className="text-ink-soft"> · repeat {run.repetition_index ?? 1}</span>
                              )}{" "}
                              <SimulatedTag run={run} />
                            </p>
                            <div className="sm:col-span-6">
                              <PassBar passed={c.passed} total={c.total} />
                            </div>
                            <p className="text-sm tabular-nums text-ink-soft sm:col-span-2 sm:text-right">
                              <span className="text-ink">{c.passed}</span>/{c.total} passed
                            </p>
                          </Link>
                        </li>
                      );
                    })}
                  </ul>
                )}
                {runList.length >= 2 && (
                  <Link to="/analysis?tab=comparison" className="mt-4 inline-block text-sm text-ink-soft hover:text-ink">
                    Compare runs →
                  </Link>
                )}
              </Section>

              {questions.length > 0 && (
                <Section title={`Questions (${questions.length})`} hint="What the model is asked, and the answer counted as correct.">
                  <ul className="divide-y divide-line">
                    {shownQuestions.map((q) => (
                      <li key={q.id} className="grid gap-1 py-2.5 text-sm sm:grid-cols-3 sm:gap-4">
                        <p className="sm:col-span-2">{q.input}</p>
                        <p className="text-ink-soft">
                          → {q.reference_answer ?? <span className="italic">no expected answer</span>}
                        </p>
                      </li>
                    ))}
                  </ul>
                  {questions.length > 5 && (
                    <button
                      type="button"
                      onClick={() => setShowAllQuestions(!showAllQuestions)}
                      className="mt-3 text-sm text-ink-soft hover:text-ink"
                    >
                      {showAllQuestions ? "Show fewer" : `Show all ${questions.length}`}
                    </button>
                  )}
                </Section>
              )}

              <SetupDetails experiment={experiment} />
            </div>
          </>
        )}
      </DataStateView>
    </div>
  );
}
