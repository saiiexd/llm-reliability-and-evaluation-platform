import { useState } from "react";
import { Link } from "react-router-dom";
import { FixtureNotice } from "../components/FixtureNotice";
import { Card, SectionHeading } from "../components/Card";
import { Tag } from "../components/Tag";
import { useComparison, useRuns } from "../data/useData";
import { DataStateView } from "../components/DataStateView";
import type { ComparisonCompatibility } from "../types/domain";

export function ExperimentComparisonPage() {
  const { data: allRuns } = useRuns();
  const [activePreset, setActivePreset] = useState<string>("identical");
  const { data: comparison, loading, error, refetch } = useComparison(activePreset);

  const compatibilityBadge = (compat: ComparisonCompatibility) => {
    switch (compat) {
      case "identical_configuration":
        return <Tag tone="neutral">Identical Configuration</Tag>;
      case "partially_different_configuration":
        return <Tag tone="caution">Controlled Intervention</Tag>;
      case "materially_incompatible":
        return <Tag tone="negative">Materially Incompatible</Tag>;
      default:
        return <Tag tone="muted">{compat}</Tag>;
    }
  };

  return (
    <div>
      <SectionHeading
        eyebrow="Comparative analysis"
        title="Cross-Run Experiment Comparison"
        description="Side-by-side comparative evaluation across experimental conditions, assessing methodological validity, parameter sensitivity, and reliability patterns."
      />
      <FixtureNotice />

      {/* Preset Selector */}
      <div className="mb-6 flex flex-wrap gap-2">
        <button
          type="button"
          onClick={() => setActivePreset("identical")}
          className={`border px-3 py-1.5 text-xs font-medium transition-colors ${
            activePreset === "identical"
              ? "border-ink bg-ink text-paper"
              : "border-line bg-paper text-ink hover:border-ink"
          }`}
        >
          Repeated Baseline Trials (Identical Config)
        </button>
        <button
          type="button"
          onClick={() => setActivePreset("intervention")}
          className={`border px-3 py-1.5 text-xs font-medium transition-colors ${
            activePreset === "intervention"
              ? "border-ink bg-ink text-paper"
              : "border-line bg-paper text-ink hover:border-ink"
          }`}
        >
          Top-K=1 vs Top-K=2 (Controlled Intervention)
        </button>
        <button
          type="button"
          onClick={() => setActivePreset("incompatible")}
          className={`border px-3 py-1.5 text-xs font-medium transition-colors ${
            activePreset === "incompatible"
              ? "border-ink bg-ink text-paper"
              : "border-line bg-paper text-ink hover:border-ink"
          }`}
        >
          RAG vs Plain Non-RAG (Materially Incompatible)
        </button>
      </div>

      <DataStateView loading={loading} error={error} onRetry={refetch}>
        {!comparison ? (
          <div className="border border-line bg-paper p-6 text-center text-xs text-ink-soft">
            No comparison data available for this selection.
          </div>
        ) : (
          <div className="space-y-6">
            {/* Compatibility Overview Banner */}
            <div className="border border-line bg-paper p-5">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <div className="font-mono text-xs uppercase text-ink-soft">
                    Methodological Compatibility
                  </div>
                  <div className="mt-1 flex items-center gap-2">
                    <span className="font-serif text-lg font-medium">
                      {comparison.compatibility === "identical_configuration" &&
                        "Repeated Trials Under Identical Setup"}
                      {comparison.compatibility === "partially_different_configuration" &&
                        "Controlled Experimental Intervention"}
                      {comparison.compatibility === "materially_incompatible" &&
                        "Materially Incompatible Runs"}
                    </span>
                    {compatibilityBadge(comparison.compatibility)}
                  </div>
                </div>

                <div className="font-mono text-xs text-ink-soft">
                  Comparing {comparison.source_run_ids.length} runs:{" "}
                  {comparison.source_run_ids.map((rId, i) => (
                    <span key={rId}>
                      {i > 0 && " vs "}
                      <Link to={`/runs/${rId}`} className="underline">
                        {rId.slice(0, 12)}
                      </Link>
                    </span>
                  ))}
                </div>
              </div>

              {comparison.compatibility_reasons.length > 0 && (
                <div className="mt-4 border-t border-line pt-3">
                  <div className="font-mono text-xs uppercase text-ink-soft">
                    Compatibility Findings & Warnings
                  </div>
                  <ul className="mt-1 list-disc space-y-1 pl-4 text-xs text-ink-soft">
                    {comparison.compatibility_reasons.map((reason, idx) => (
                      <li key={idx}>{reason}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            {/* Configuration Differences */}
            {Object.keys(comparison.configuration_diffs).length > 0 && (
              <Card>
                <SectionHeading
                  eyebrow="Configuration deltas"
                  title="Experimental Parameter Diffs"
                  description="Configuration parameters that differ between the compared runs."
                />
                <div className="space-y-2 font-mono text-xs">
                  {Object.entries(comparison.configuration_diffs).map(([param, vals]) => (
                    <div key={param} className="border-b border-line pb-2 last:border-0">
                      <span className="font-semibold text-ink">{param}: </span>
                      {Object.entries(vals as Record<string, unknown>).map(([rId, v]) => (
                        <span key={rId} className="mr-3 text-ink-soft">
                          <code className="bg-paper-dim px-1">{rId.slice(0, 8)}</code> ={" "}
                          <span className="text-ink">{JSON.stringify(v)}</span>
                        </span>
                      ))}
                    </div>
                  ))}
                </div>
              </Card>
            )}

            {/* Side-by-Side Evaluator Metrics */}
            <Card>
              <SectionHeading
                eyebrow="Evaluation comparison"
                title="Metric Outcomes per Evaluator"
                description="Pass rates and mean scores recorded for each evaluator across runs."
              />
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs font-mono">
                  <thead>
                    <tr className="border-b border-line text-ink-soft">
                      <th className="py-2 pr-4 font-normal font-sans">Evaluator / Metric</th>
                      {comparison.source_run_ids.map((rId) => {
                        const runObj = allRuns?.find((r) => r.run_id === rId);
                        return (
                          <th key={rId} className="py-2 pr-4 font-normal">
                            <div>{rId.slice(0, 10)}</div>
                            <div className="text-[10px] text-ink-soft">
                              {runObj?.model_config?.model_id ?? "Run"}
                            </div>
                          </th>
                        );
                      })}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-line">
                    {Object.entries(comparison.evaluation_comparisons).map(([name, data]) => {
                      const typedData = data;
                      return (
                        <tr key={name}>
                          <td className="py-2 pr-4">
                            <div className="font-medium text-ink">{name}</div>
                          </td>
                          {comparison.source_run_ids.map((rId) => {
                            const pr = typedData.pass_rates[rId];
                            const ms = typedData.score_means[rId];
                            return (
                              <td key={rId} className="py-2 pr-4">
                                <div>
                                  Pass:{" "}
                                  <span className="font-semibold text-ink">
                                    {pr !== null && pr !== undefined
                                      ? `${(pr * 100).toFixed(1)}%`
                                      : "—"}
                                  </span>
                                </div>
                                {ms !== null && ms !== undefined && (
                                  <div className="text-[10px] text-ink-soft">
                                    Mean score: {ms.toFixed(3)}
                                  </div>
                                )}
                              </td>
                            );
                          })}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Card>

            {/* Retrieval Metrics Side-by-Side */}
            <div className="grid gap-6 lg:grid-cols-2">
              <Card>
                <SectionHeading
                  eyebrow="Retrieval comparison"
                  title="RAG Retrieval Metrics"
                  description="Retrieval quality across runs under controlled K settings."
                />
                {Object.keys(comparison.retrieval_comparisons).length === 0 ? (
                  <p className="text-xs text-ink-soft">No retrieval evaluations to compare.</p>
                ) : (
                  <table className="w-full text-left text-xs font-mono">
                    <thead>
                      <tr className="border-b border-line text-ink-soft">
                        <th className="py-1.5 pr-4 font-normal font-sans">Metric</th>
                        {comparison.source_run_ids.map((rId) => (
                          <th key={rId} className="py-1.5 pr-4 font-normal">
                            {rId.slice(0, 8)}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-line">
                      {Object.entries(comparison.retrieval_comparisons).map(([metric, scores]) => (
                        <tr key={metric}>
                          <td className="py-1.5 pr-4 text-ink-soft">{metric}</td>
                          {comparison.source_run_ids.map((rId) => (
                            <td key={rId} className="py-1.5 pr-4">
                              {scores[rId] !== null && scores[rId] !== undefined
                                ? scores[rId]?.toFixed(3)
                                : "—"}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </Card>

              <Card>
                <SectionHeading
                  eyebrow="Input robustness"
                  title="Consistency rate under perturbation"
                  description="Fraction of test cases whose diagnostic outcome remained unchanged."
                />
                <table className="w-full text-left text-xs font-mono">
                  <thead>
                    <tr className="border-b border-line text-ink-soft">
                      <th className="py-1.5 pr-4 font-normal font-sans">Run</th>
                      <th className="py-1.5 font-normal">Consistency Rate</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-line">
                    {comparison.source_run_ids.map((rId) => {
                      const rate = comparison.robustness_comparisons[rId];
                      return (
                        <tr key={rId}>
                          <td className="py-1.5 pr-4 font-medium">
                            <Link to={`/runs/${rId}`} className="underline">
                              {rId.slice(0, 10)}
                            </Link>
                          </td>
                          <td className="py-1.5">
                            {rate !== null && rate !== undefined
                              ? `${(rate * 100).toFixed(1)}%`
                              : "Not evaluated"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </Card>
            </div>
          </div>
        )}
      </DataStateView>
    </div>
  );
}
