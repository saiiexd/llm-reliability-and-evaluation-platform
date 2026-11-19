import { useState } from "react";
import { Link } from "react-router-dom";
import { FixtureNotice } from "../components/FixtureNotice";
import { Card, SectionHeading } from "../components/Card";
import { Tag } from "../components/Tag";
import { usePerformanceAnalytics, useRuns } from "../data/useData";
import { DataStateView } from "../components/DataStateView";

export function PerformanceAnalyticsPage() {
  const { data: runs, loading: runsLoading } = useRuns();
  const [selectedRunId, setSelectedRunId] = useState<string>("");

  const effectiveRunId = selectedRunId || (runs && runs[0]?.run_id) || "";
  const { data: perf, loading: perfLoading, error, refetch } = usePerformanceAnalytics(effectiveRunId);

  const runList = runs || [];

  return (
    <div>
      <SectionHeading
        eyebrow="Instrumentation & Latency"
        title="Execution Performance & Telemetry"
        description="Measured execution durations, latency instrumentation, and explicit reporting of telemetry boundaries. Distinguishes observed execution characteristics from uninstrumented metrics."
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

      <DataStateView loading={runsLoading || perfLoading} error={error} onRetry={refetch}>
        {!perf ? (
          <Card>
            <p className="text-sm text-ink-soft">No performance telemetry recorded for this run.</p>
          </Card>
        ) : (
          <div className="space-y-6">
            {/* Measured Durations */}
            {(() => {
              const durationsInstrumented = perf.per_case_duration_seconds_mean !== null;
              return (
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <Card>
                    <div className="text-xs text-ink-soft">Wall-clock duration</div>
                    <div className="mt-1 font-mono text-xl font-medium">
                      {perf.wall_clock_duration_seconds !== null
                        ? `${perf.wall_clock_duration_seconds.toFixed(2)}s`
                        : "Unavailable"}
                    </div>
                    <div className="mt-1 text-xs text-ink-soft">From UTC start/finish timestamps</div>
                  </Card>

                  <Card>
                    <div className="text-xs text-ink-soft">Mean case latency</div>
                    <div className="mt-1 font-mono text-xl font-medium">
                      {perf.per_case_duration_seconds_mean !== null
                        ? `${(perf.per_case_duration_seconds_mean * 1000).toFixed(0)}ms`
                        : "Uninstrumented"}
                    </div>
                    <div className="mt-1 text-xs text-ink-soft">Arithmetic mean across cases</div>
                  </Card>

                  <Card>
                    <div className="text-xs text-ink-soft">Instrumentation status</div>
                    <div className="mt-2">
                      <Tag tone={durationsInstrumented ? "neutral" : "muted"}>
                        {durationsInstrumented ? "Instrumented" : "Uninstrumented"}
                      </Tag>
                    </div>
                    <div className="mt-1 text-xs text-ink-soft">Case-level timing capture</div>
                  </Card>

                  <Card>
                    <div className="text-xs text-ink-soft">Token usage telemetry</div>
                    <div className="mt-2">
                      <Tag tone={perf.tokens_available ? "neutral" : "muted"}>
                        {perf.tokens_available ? "Reported" : "Unavailable"}
                      </Tag>
                    </div>
                    <div className="mt-1 text-xs text-ink-soft">Provider token counts</div>
                  </Card>
                </div>
              );
            })()}

            {/* Telemetry Availability & Explicit Limitations */}
            <div className="grid gap-6 lg:grid-cols-2">
              <Card>
                <SectionHeading
                  eyebrow="Coverage disclosure"
                  title="Telemetry Boundaries & Data Availability"
                  description="Honest reporting of which execution measurements were captured vs unavailable."
                />
                <dl className="space-y-3 text-xs">
                  <div className="flex items-center justify-between border-b border-line pb-2">
                    <dt className="text-ink">Execution Timestamps</dt>
                    <dd>
                      <Tag tone="neutral">Available (UTC ISO-8601)</Tag>
                    </dd>
                  </div>
                  <div className="flex items-center justify-between border-b border-line pb-2">
                    <dt className="text-ink">Per-case Latencies</dt>
                    <dd>
                      <Tag tone={perf.per_case_duration_seconds_mean !== null ? "neutral" : "muted"}>
                        {perf.per_case_duration_seconds_mean !== null ? "Instrumented" : "Unavailable in mock"}
                      </Tag>
                    </dd>
                  </div>
                  <div className="flex items-center justify-between border-b border-line pb-2">
                    <dt className="text-ink">Token Usage Telemetry</dt>
                    <dd>
                      <Tag tone={perf.tokens_available ? "neutral" : "muted"}>
                        {perf.tokens_available ? "Reported" : "Not Provided"}
                      </Tag>
                    </dd>
                  </div>
                  <div className="flex items-center justify-between pb-1">
                    <dt className="text-ink">Financial Cost Modeling</dt>
                    <dd>
                      <Tag tone="muted">Not Calculated</Tag>
                    </dd>
                  </div>
                </dl>
              </Card>

              <Card>
                <SectionHeading
                  eyebrow="Methodological note"
                  title="Interpretation Guidelines"
                  description="Explicit architectural rules for analyzing execution telemetry."
                />
                <div className="space-y-3 text-xs text-ink-soft leading-relaxed">
                  <p>
                    <strong className="text-ink">Wall-clock vs Latency Sum:</strong> The wall-clock
                    duration measures total elapsed real time, which includes framework overhead,
                    persistence serialization, and file I/O. The test case duration sum captures
                    only the isolated execution loop.
                  </p>
                  <p>
                    <strong className="text-ink">Pricing & Cost Exclusion:</strong> The platform does
                    not calculate financial costs or vendor token prices. Token pricing models vary
                    continuously across providers, contract tiers, and caching structures. Any
                    pricing figures would be synthetic and non-reproducible.
                  </p>
                  <p className="border-t border-line pt-2 text-ink italic">
                    Performance measurements describe execution characteristics only; they are
                    never used as a proxy for answer quality or reliability.
                  </p>
                </div>
              </Card>
            </div>

            {/* Telemetry Notes */}
            {perf.notes.length > 0 && (
              <Card>
                <div className="font-mono text-xs uppercase tracking-wide text-ink-soft">
                  Instrumentation Notes
                </div>
                <ul className="mt-2 list-disc space-y-1 pl-4 text-xs text-ink-soft">
                  {perf.notes.map((note, idx) => (
                    <li key={idx}>{note}</li>
                  ))}
                </ul>
              </Card>
            )}

            {/* Backlink */}
            <div className="pt-2">
              <Link
                to={`/runs/${effectiveRunId}`}
                className="font-mono text-xs text-ink underline"
              >
                &larr; View Full Run Detail ({effectiveRunId})
              </Link>
            </div>
          </div>
        )}
      </DataStateView>
    </div>
  );
}
