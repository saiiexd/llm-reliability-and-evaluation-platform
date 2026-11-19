import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { FixtureNotice } from "../components/FixtureNotice";
import { Card, SectionHeading } from "../components/Card";
import { EmptyState } from "../components/EmptyState";
import { Tag } from "../components/Tag";
import { toneForRunStatus } from "../lib/tone";
import { formatTimestamp, shortFingerprint } from "../lib/derived";
import { useExperiment, useRuns } from "../data/useData";
import { useConnection } from "../context/ConnectionContext";
import { DataStateView } from "../components/DataStateView";
import type { RunExecutionResponse } from "../api/client";
import type { BenchmarkExecutionResponse, ProviderStatus } from "../types/domain";

const ALLOWLISTED_ADAPTERS = [
  {
    id: "rag-pipeline-v1",
    label: "RAG Pipeline Top-K=1",
    isExternal: false,
    detail: "Local embedding retriever + canned generator",
  },
  {
    id: "rag-pipeline-topk-2",
    label: "RAG Pipeline Top-K=2",
    isExternal: false,
    detail: "Local embedding retriever top-2",
  },
  {
    id: "mock-adapter-v1",
    label: "Deterministic Mock",
    isExternal: false,
    detail: "Offline canned responses",
  },
  {
    id: "echo-adapter",
    label: "Echo Adapter",
    isExternal: false,
    detail: "Deterministic input echo",
  },
  {
    id: "openai-gpt-4o-mini",
    label: "OpenAI GPT-4o Mini (Live)",
    isExternal: true,
    detail: "Official OpenAI API live inference",
  },
  {
    id: "openai-gpt-4o",
    label: "OpenAI GPT-4o (Live)",
    isExternal: true,
    detail: "Official OpenAI API flagship live inference",
  },
  {
    id: "openai-custom",
    label: "OpenAI-Compatible Custom (Live)",
    isExternal: true,
    detail: "Local/remote vLLM, Ollama, or LM Studio endpoint",
  },
];

export function ExperimentDetailPage() {
  const { experimentId = "" } = useParams<{ experimentId: string }>();
  const { data: experiment, loading: expLoading, error: expError, refetch: refetchExp } = useExperiment(experimentId);
  const { data: runs, loading: runsLoading, error: runsError, refetch: refetchRuns } = useRuns(experimentId);
  const { dataMode, executeRun, executeBenchmark, client } = useConnection();

  const [execMode, setExecMode] = useState<"single" | "benchmark">("single");
  const [selectedAdapter, setSelectedAdapter] = useState<string>("rag-pipeline-v1");
  const [selectedBenchmarkAdapters, setSelectedBenchmarkAdapters] = useState<string[]>([
    "mock-adapter-v1",
    "echo-adapter",
  ]);
  const [repetitions, setRepetitions] = useState<number>(1);
  const [confirmedExternal, setConfirmedExternal] = useState<boolean>(false);

  const [executing, setExecuting] = useState(false);
  const [execResult, setExecResult] = useState<RunExecutionResponse | null>(null);
  const [benchResult, setBenchResult] = useState<BenchmarkExecutionResponse | null>(null);
  const [execError, setExecError] = useState<string | null>(null);
  const [providers, setProviders] = useState<ProviderStatus[]>([]);

  // Fetch provider readiness when in live mode
  useEffect(() => {
    if (dataMode === "live") {
      client
        .listProviders()
        .then((res) => setProviders(res.providers))
        .catch(() => {
          // Gracefully continue if providers endpoint is unavailable
        });
    }
  }, [client, dataMode]);

  const activeAdapterObj = ALLOWLISTED_ADAPTERS.find((a) => a.id === selectedAdapter);
  const singleRequiresExternal = activeAdapterObj?.isExternal ?? false;
  const benchmarkHasExternal = selectedBenchmarkAdapters.some(
    (id) => ALLOWLISTED_ADAPTERS.find((a) => a.id === id)?.isExternal,
  );

  const openaiProvider = providers.find((p) => p.provider_id === "openai");
  const isOpenAiReady = openaiProvider?.is_configured ?? false;

  const handleSingleExecute = async () => {
    setExecuting(true);
    setExecResult(null);
    setBenchResult(null);
    setExecError(null);
    try {
      const result = await executeRun(experimentId, selectedAdapter, confirmedExternal);
      setExecResult(result);
      void refetchRuns();
    } catch (err: unknown) {
      setExecError(err instanceof Error ? err.message : String(err));
    } finally {
      setExecuting(false);
    }
  };

  const handleBenchmarkExecute = async () => {
    if (selectedBenchmarkAdapters.length === 0) {
      setExecError("Please select at least one adapter for the benchmark.");
      return;
    }
    setExecuting(true);
    setExecResult(null);
    setBenchResult(null);
    setExecError(null);
    try {
      const result = await executeBenchmark({
        experiment_id: experimentId,
        adapter_ids: selectedBenchmarkAdapters,
        repetitions,
        allow_external_inference: confirmedExternal,
      });
      setBenchResult(result);
      void refetchRuns();
    } catch (err: unknown) {
      setExecError(err instanceof Error ? err.message : String(err));
    } finally {
      setExecuting(false);
    }
  };

  const toggleBenchmarkAdapter = (adapterId: string) => {
    setSelectedBenchmarkAdapters((prev) =>
      prev.includes(adapterId) ? prev.filter((id) => id !== adapterId) : [...prev, adapterId],
    );
  };

  const loading = expLoading || runsLoading;
  const error = expError || runsError;
  const runList = runs || [];

  return (
    <div>
      <DataStateView
        loading={loading}
        error={error}
        onRetry={() => {
          void refetchExp();
          void refetchRuns();
        }}
      >
        {!experiment ? (
          <EmptyState
            title="Experiment not found."
            detail={`No experiment with id "${experimentId}".`}
          />
        ) : (
          <>
            <SectionHeading eyebrow="Experiment definition" title={experiment.name} />
            <FixtureNotice />
            {experiment.description && (
              <p className="mb-6 max-w-2xl text-sm">{experiment.description}</p>
            )}

            {/* Controlled Execution & Benchmarking Section */}
            <div className="mb-6 border border-line bg-paper p-5">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between border-b border-line pb-4 mb-4">
                <div>
                  <h4 className="font-display text-base font-semibold">Model Execution & Controlled Benchmarking</h4>
                  <p className="mt-0.5 text-xs text-ink-soft">
                    Execute individual allowlisted models or launch controlled multi-model benchmarks with variance repetitions.
                  </p>
                </div>
                {/* Provider Readiness Badges */}
                {dataMode === "live" && (
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-xs text-ink-soft">OpenAI Readiness:</span>
                    <Tag tone={isOpenAiReady ? "neutral" : "negative"}>
                      {isOpenAiReady ? "Configured" : "Key Missing in Server"}
                    </Tag>
                  </div>
                )}
              </div>

              {/* Mode Toggle */}
              <div className="mb-4 flex items-center gap-2 border-b border-line pb-3">
                <button
                  type="button"
                  onClick={() => setExecMode("single")}
                  className={`px-3 py-1 font-mono text-xs font-medium transition-colors ${
                    execMode === "single"
                      ? "border-b-2 border-ink text-ink font-semibold"
                      : "text-ink-soft hover:text-ink"
                  }`}
                >
                  Single Run Execution
                </button>
                <button
                  type="button"
                  onClick={() => setExecMode("benchmark")}
                  className={`px-3 py-1 font-mono text-xs font-medium transition-colors ${
                    execMode === "benchmark"
                      ? "border-b-2 border-ink text-ink font-semibold"
                      : "text-ink-soft hover:text-ink"
                  }`}
                >
                  Multi-Model Benchmark Matrix
                </button>
              </div>

              {dataMode !== "live" && (
                <div className="mb-4 border border-amber-500/40 bg-amber-950/30 p-3 text-xs text-amber-200">
                  Interactive execution requires <strong>Live API Mode</strong>. Switch to Live API to run live evaluations.
                </div>
              )}

              {/* Single Model Execution Controls */}
              {execMode === "single" && (
                <div className="space-y-4">
                  <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
                    <label htmlFor="adapter-select" className="font-mono text-xs text-ink-soft w-28">
                      Model Adapter:
                    </label>
                    <select
                      id="adapter-select"
                      value={selectedAdapter}
                      onChange={(e) => setSelectedAdapter(e.target.value)}
                      disabled={executing || dataMode !== "live"}
                      className="border border-line bg-paper px-3 py-1.5 font-mono text-xs text-ink focus:border-ink focus:outline-none disabled:opacity-50 flex-1 max-w-md"
                    >
                      {ALLOWLISTED_ADAPTERS.map((a) => (
                        <option key={a.id} value={a.id}>
                          {a.label} — {a.detail}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Safety Guard Warning for External Inference */}
                  {singleRequiresExternal && (
                    <div className="border border-amber-500/40 bg-amber-950/30 p-3 text-xs text-amber-200">
                      <div className="flex items-start gap-2">
                        <span className="font-bold text-amber-400">⚠ Live External Inference:</span>
                        <div className="space-y-1.5">
                          <p>
                            This adapter issues live outbound calls to an external LLM API. API credits or usage fees may apply.
                            Credentials must be configured on the server via <code className="bg-amber-900/50 px-1 font-mono text-amber-200">OPENAI_API_KEY</code>.
                          </p>
                          <label className="flex items-center gap-2 font-medium cursor-pointer text-amber-200">
                            <input
                              type="checkbox"
                              checked={confirmedExternal}
                              onChange={(e) => setConfirmedExternal(e.target.checked)}
                              disabled={executing}
                              className="accent-ink"
                            />
                            I understand and explicitly permit external model inference (allow_external_inference)
                          </label>
                        </div>
                      </div>
                    </div>
                  )}

                  <div>
                    <button
                      type="button"
                      onClick={handleSingleExecute}
                      disabled={executing || dataMode !== "live" || (singleRequiresExternal && !confirmedExternal)}
                      className="bg-ink px-4 py-2 font-mono text-xs font-medium text-paper hover:bg-ink-soft disabled:cursor-not-allowed disabled:opacity-40"
                    >
                      {executing ? "Executing Run..." : "Execute Single Run"}
                    </button>
                  </div>
                </div>
              )}

              {/* Multi-Model Benchmark Matrix Controls */}
              {execMode === "benchmark" && (
                <div className="space-y-4">
                  <div>
                    <label className="block font-mono text-xs uppercase tracking-wide text-ink-soft mb-2">
                      Select Model Adapters to Compare:
                    </label>
                    <div className="grid gap-2 sm:grid-cols-2">
                      {ALLOWLISTED_ADAPTERS.map((a) => (
                        <label
                          key={a.id}
                          className={`flex items-start gap-2 border p-2 text-xs cursor-pointer transition-colors ${
                            selectedBenchmarkAdapters.includes(a.id)
                              ? "border-ink bg-paper-dim"
                              : "border-line bg-paper"
                          }`}
                        >
                          <input
                            type="checkbox"
                            checked={selectedBenchmarkAdapters.includes(a.id)}
                            onChange={() => toggleBenchmarkAdapter(a.id)}
                            disabled={executing}
                            className="mt-0.5 accent-ink"
                          />
                          <div>
                            <div className="font-mono font-medium text-ink">{a.label}</div>
                            <div className="text-ink-soft text-[11px]">{a.detail}</div>
                          </div>
                        </label>
                      ))}
                    </div>
                  </div>

                  <div className="flex items-center gap-4">
                    <label htmlFor="repetition-input" className="font-mono text-xs text-ink-soft">
                      Repetitions per Model:
                    </label>
                    <input
                      id="repetition-input"
                      type="number"
                      min={1}
                      max={5}
                      value={repetitions}
                      onChange={(e) => setRepetitions(Math.max(1, Math.min(5, parseInt(e.target.value) || 1)))}
                      disabled={executing || dataMode !== "live"}
                      className="w-20 border border-line bg-paper px-2 py-1 font-mono text-xs text-ink focus:border-ink focus:outline-none"
                    />
                    <span className="text-[11px] text-ink-soft italic">
                      Repetitions measure stochastic variance across trials without collapsing runs into a synthetic score.
                    </span>
                  </div>

                  {/* Safety Guard for Benchmark */}
                  {benchmarkHasExternal && (
                    <div className="border border-amber-500/40 bg-amber-950/30 p-3 text-xs text-amber-200">
                      <div className="flex items-start gap-2">
                        <span className="font-bold text-amber-400">⚠ External Inference in Benchmark:</span>
                        <div className="space-y-1.5">
                          <p>
                            One or more selected benchmark models require live outbound API calls.
                            Ensure your server has API credentials configured.
                          </p>
                          <label className="flex items-center gap-2 font-medium cursor-pointer text-amber-200">
                            <input
                              type="checkbox"
                              checked={confirmedExternal}
                              onChange={(e) => setConfirmedExternal(e.target.checked)}
                              disabled={executing}
                              className="accent-ink"
                            />
                            I understand and explicitly permit external model inference for this benchmark
                          </label>
                        </div>
                      </div>
                    </div>
                  )}

                  <div>
                    <button
                      type="button"
                      onClick={handleBenchmarkExecute}
                      disabled={
                        executing ||
                        dataMode !== "live" ||
                        selectedBenchmarkAdapters.length === 0 ||
                        (benchmarkHasExternal && !confirmedExternal)
                      }
                      className="bg-ink px-4 py-2 font-mono text-xs font-medium text-paper hover:bg-ink-soft disabled:cursor-not-allowed disabled:opacity-40"
                    >
                      {executing
                        ? "Executing Benchmark..."
                        : `Launch Benchmark (${selectedBenchmarkAdapters.length} models × ${repetitions} repeats = ${
                            selectedBenchmarkAdapters.length * repetitions
                          } runs)`}
                    </button>
                  </div>
                </div>
              )}

              {/* Single Run Result */}
              {execResult && (
                <div className="mt-4 border border-emerald-500/40 bg-emerald-950/30 p-3 text-xs text-emerald-200">
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="font-semibold text-emerald-300">Run completed successfully: </span>
                      <span className="font-mono">{execResult.run_id}</span> ({execResult.duration_seconds.toFixed(2)}s,{" "}
                      {execResult.num_succeeded}/{execResult.test_cases_count} succeeded)
                    </div>
                    <Link
                      to={`/results/${execResult.run_id}`}
                      className="font-medium text-emerald-400 underline hover:text-emerald-300 font-mono"
                    >
                      Inspect Evaluation Results →
                    </Link>
                  </div>
                </div>
              )}

              {/* Benchmark Result */}
              {benchResult && (
                <div className="mt-4 border border-emerald-500/40 bg-emerald-950/30 p-4 text-xs text-emerald-200 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-emerald-300">
                      Benchmark Completed ({benchResult.total_runs} runs executed):
                    </span>
                    <span className="font-mono font-medium">{benchResult.benchmark_id}</span>
                  </div>
                  <div className="overflow-x-auto border border-line bg-paper">
                    <table className="w-full text-left font-mono text-xs">
                      <thead>
                        <tr className="border-b border-line bg-paper-dim">
                          <th className="p-2">Run ID</th>
                          <th className="p-2">Status</th>
                          <th className="p-2">Duration</th>
                          <th className="p-2">Succeeded / Total</th>
                          <th className="p-2">Action</th>
                        </tr>
                      </thead>
                      <tbody>
                        {benchResult.runs.map((r) => (
                          <tr key={r.run_id} className="border-b border-line last:border-0">
                            <td className="p-2 font-medium">{r.run_id}</td>
                            <td className="p-2">{r.status}</td>
                            <td className="p-2">{r.duration_seconds ? `${r.duration_seconds.toFixed(2)}s` : "n/a"}</td>
                            <td className="p-2">{r.num_succeeded} / {r.num_succeeded + r.num_failed}</td>
                            <td className="p-2">
                              <Link
                                to={`/results/${r.run_id}`}
                                className="underline text-emerald-400 hover:text-emerald-300"
                              >
                                View Results →
                              </Link>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Execution Error Message */}
              {execError && (
                <div className="mt-4 border border-red-500/40 bg-red-950/30 p-3 text-xs text-red-200">
                  <span className="font-semibold text-red-300">Execution failed: </span>
                  {execError}
                </div>
              )}
            </div>

            <div className="grid gap-6 lg:grid-cols-2">
              <Card>
                <h3 className="mb-2 font-display text-lg">Reproducibility</h3>
                <dl className="space-y-2 text-sm">
                  <Row label="Experiment ID" value={<span className="font-mono">{experiment.experiment_id}</span>} />
                  <Row
                    label="Config fingerprint"
                    value={<span className="font-mono">{shortFingerprint(experiment.config_fingerprint)}</span>}
                  />
                  <Row label="Created" value={formatTimestamp(experiment.created_at)} />
                </dl>
              </Card>

              <Card>
                <h3 className="mb-2 font-display text-lg">Dataset</h3>
                <dl className="space-y-2 text-sm">
                  <Row label="Name" value={experiment.dataset_version?.dataset_name || "N/A"} />
                  <Row
                    label="Version ID"
                    value={
                      <span className="font-mono">
                        {shortFingerprint(experiment.dataset_version?.version_id || "")}
                      </span>
                    }
                  />
                  <Row label="Test cases" value={experiment.dataset_version?.test_case_count ?? experiment.dataset_version?.test_cases?.length ?? experiment.test_case_count ?? 0} />
                </dl>
              </Card>

              <Card>
                <h3 className="mb-2 font-display text-lg">Model / application configuration</h3>
                <dl className="space-y-2 text-sm">
                  <Row label="Model ID" value={<span className="font-mono">{experiment.model_config?.model_id || "N/A"}</span>} />
                  <Row
                    label="Temperature"
                    value={
                      experiment.model_config?.temperature !== null &&
                      experiment.model_config?.temperature !== undefined
                        ? String(experiment.model_config.temperature)
                        : "not specified"
                    }
                  />
                  <Row
                    label="Max tokens"
                    value={
                      experiment.model_config?.max_tokens !== null &&
                      experiment.model_config?.max_tokens !== undefined
                        ? String(experiment.model_config.max_tokens)
                        : "not specified"
                    }
                  />
                </dl>
                {Boolean(experiment.model_config?.extra_params?.rag) && (
                  <div className="mt-3 border-t border-line pt-3">
                    <p className="mb-1 font-mono text-xs uppercase text-ink-soft">RAG configuration</p>
                    <dl className="space-y-1 text-sm">
                      {(() => {
                        const ragConfig = (experiment.model_config.extra_params.rag || {}) as Record<string, unknown>;
                        return (
                          <>
                            <Row label="Embedding model" value={<span className="font-mono">{String(ragConfig.embedding_model_id ?? "")}</span>} />
                            <Row label="Chunk size / overlap" value={`${String(ragConfig.chunk_size ?? "")} / ${String(ragConfig.chunk_overlap ?? "")}`} />
                            <Row label="Top K" value={String(ragConfig.top_k ?? "")} />
                          </>
                        );
                      })()}
                    </dl>
                  </div>
                )}
              </Card>

              <Card>
                <h3 className="mb-2 font-display text-lg">Evaluators ({experiment.evaluator_configs?.length ?? 0})</h3>
                <ul className="space-y-2 text-sm">
                  {experiment.evaluator_configs?.map((evaluator) => (
                    <li key={evaluator.evaluator_name} className="border-b border-line pb-2 last:border-0">
                      <div className="font-mono font-medium">{evaluator.evaluator_name}</div>
                      {Object.keys(evaluator.parameters || {}).length > 0 && (
                        <pre className="mt-1 font-mono text-xs text-ink-soft">
                          {JSON.stringify(evaluator.parameters, null, 2)}
                        </pre>
                      )}
                    </li>
                  ))}
                </ul>
              </Card>
            </div>

            <div className="mt-8">
              <Card>
                <div className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                  <SectionHeading eyebrow="Execution history" title={`Runs (${runList.length})`} />
                  <Link
                    to="/analysis?tab=comparison"
                    className="font-mono text-xs text-ink underline hover:text-ink-soft"
                  >
                    Compare Runs Side-by-Side &rarr;
                  </Link>
                </div>
                {runList.length === 0 ? (
                  <EmptyState title="No runs executed yet for this experiment." />
                ) : (
                  <div className="overflow-x-auto border border-line">
                    <table className="w-full min-w-[640px] border-collapse text-sm">
                      <thead>
                        <tr className="border-b border-line bg-paper-dim text-left font-mono text-xs uppercase">
                          <th className="px-3 py-2">Run ID</th>
                          <th className="px-3 py-2">Status</th>
                          <th className="px-3 py-2">Model ID</th>
                          <th className="px-3 py-2">Started</th>
                          <th className="px-3 py-2">Duration</th>
                          <th className="px-3 py-2">Benchmark / Rep</th>
                        </tr>
                      </thead>
                      <tbody>
                        {runList.map((run) => (
                          <tr key={run.run_id} className="border-b border-line last:border-0 hover:bg-paper-dim">
                            <td className="px-3 py-2 font-mono text-xs">
                              <Link to={`/results/${run.run_id}`} className="underline">
                                {shortFingerprint(run.run_id, 16)}
                              </Link>
                            </td>
                            <td className="px-3 py-2">
                              <Tag tone={toneForRunStatus(run.status)}>{run.status}</Tag>
                            </td>
                            <td className="px-3 py-2 font-mono text-xs text-ink-soft">
                              {run.model_config?.model_id || "n/a"}
                            </td>
                            <td className="px-3 py-2 text-xs">{formatTimestamp(run.started_at)}</td>
                            <td className="px-3 py-2 font-mono text-xs">
                              {run.duration_seconds !== null && run.duration_seconds !== undefined
                                ? `${run.duration_seconds.toFixed(2)}s`
                                : "n/a"}
                            </td>
                            <td className="px-3 py-2 font-mono text-xs text-ink-soft">
                              {run.benchmark_id ? (
                                <span>
                                  {shortFingerprint(run.benchmark_id, 8)} (rep #{run.repetition_index ?? 1})
                                </span>
                              ) : (
                                "—"
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </Card>
            </div>
          </>
        )}
      </DataStateView>
    </div>
  );
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between border-b border-line py-1 last:border-0">
      <dt className="text-ink-soft">{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}
