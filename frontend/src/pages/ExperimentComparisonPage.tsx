import { useState } from "react";
import { Link } from "react-router-dom";
import { useComparison, useRuns } from "../data/useData";
import { DataStateView } from "../components/DataStateView";
import { EmptyState } from "../components/EmptyState";
import { evaluatorLabel, formatTimestamp } from "../lib/derived";
import type { ComparisonCompatibility } from "../types/domain";

const PRESETS = [
  { id: "identical", label: "Same setup, run twice" },
  { id: "intervention", label: "1 vs 2 retrieved passages" },
  { id: "incompatible", label: "RAG vs no RAG" },
];

const VERDICT: Record<ComparisonCompatibility, { title: string; body: string }> = {
  identical_configuration: {
    title: "Same setup",
    body: "Nothing was changed between these runs, so any difference in results is randomness, not a real effect.",
  },
  partially_different_configuration: {
    title: "Some settings changed",
    body: "Only the settings listed under “What changed” differ, so differences in results can be attributed to them.",
  },
  materially_incompatible: {
    title: "Not a fair comparison",
    body: "Too much differs between these runs to say what caused any difference in results.",
  },
};

const METRIC_LABEL: Record<string, string> = {
  hit_at_k_k1: "Right passage found in top 1",
  hit_at_k_k2: "Right passage found in top 2",
  recall_at_k_k1: "Share of needed passages found (top 1)",
  recall_at_k_k2: "Share of needed passages found (top 2)",
  mrr: "How high the right passage ranked (MRR)",
};

/** Flatten nested config values to dotted paths so we can diff leaf settings. */
function flatten(value: unknown, prefix: string, out: Record<string, string>) {
  if (value !== null && typeof value === "object" && !Array.isArray(value)) {
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      flatten(v, prefix ? `${prefix}.${k}` : k, out);
    }
  } else {
    out[prefix] = value === undefined || value === null ? "not set" : Array.isArray(value) ? value.join(", ") : String(value);
  }
}

/** Only the settings whose values actually differ between runs. */
function changedSettings(
  diffs: Record<string, unknown>,
  runIds: string[],
): { path: string; values: string[] }[] {
  const perRun = runIds.map((id) => {
    const flat: Record<string, string> = {};
    for (const [param, byRun] of Object.entries(diffs)) {
      flatten((byRun as Record<string, unknown> | null)?.[id], param, flat);
    }
    return flat;
  });
  const paths = new Set(perRun.flatMap((f) => Object.keys(f)));
  return [...paths]
    // The config fingerprint is a hash of all the other settings, so it adds nothing.
    .filter((p) => !/config_fingerprint/i.test(p))
    .map((path) => ({
      path,
      // Dataset version IDs are long hashes; the first 8 characters are enough to tell them apart.
      values: perRun.map((f) => {
        const v = f[path] ?? "not set";
        // Long hex hashes (versions, fingerprints): 8 characters are enough to tell them apart.
        return /^[0-9a-f]{16,}$/i.test(v) ? v.slice(0, 8) : v;
      }),
    }))
    .filter((row) => new Set(row.values).size > 1);
}

const pct = (v: number | null | undefined) => (v == null ? "—" : `${Math.round(v * 100)}%`);

export function ExperimentComparisonPage() {
  const { data: allRuns } = useRuns();
  const [preset, setPreset] = useState("identical");
  const { data: comparison, loading, error, refetch } = useComparison(preset);

  const runIds = comparison?.source_run_ids ?? [];
  const runInfo = (id: string) => allRuns?.find((r) => r.run_id === id);
  const letter = (i: number) => String.fromCharCode(65 + i);

  type Row = { label: string; values: (number | null | undefined)[]; format: (v: number | null | undefined) => string };
  const rows: Row[] = comparison
    ? [
        ...Object.entries(comparison.evaluation_comparisons).map(([name, d]) => ({
          label: `${evaluatorLabel(name)}: passed`,
          values: runIds.map((id) => d.pass_rates[id]),
          format: pct,
        })),
        ...Object.entries(comparison.retrieval_comparisons).map(([metric, scores]) => ({
          label: METRIC_LABEL[metric] ?? metric,
          values: runIds.map((id) => scores[id]),
          format: (v: number | null | undefined) => (v == null ? "—" : v.toFixed(2)),
        })),
        {
          label: "Same verdict when the input is changed",
          values: runIds.map((id) => comparison.robustness_comparisons[id]),
          format: pct,
        },
      ]
    : [];

  return (
    <div>
      <div className="mb-6 flex flex-wrap gap-2">
        {PRESETS.map((p) => (
          <button
            key={p.id}
            type="button"
            onClick={() => setPreset(p.id)}
            className={`rounded-full border px-3 py-1.5 text-sm transition-colors ${
              preset === p.id ? "border-ink bg-ink text-paper" : "border-line text-ink-soft hover:text-ink"
            }`}
          >
            {p.label}
          </button>
        ))}
      </div>

      <DataStateView loading={loading} error={error} onRetry={refetch}>
        {!comparison ? (
          <EmptyState title="No comparison available for this pair." />
        ) : (
          <div className="space-y-6">
            <section className="rounded-xl border border-line bg-paper-dim p-6">
              <p className="text-lg font-semibold">{VERDICT[comparison.compatibility]?.title ?? comparison.compatibility}</p>
              <p className="mt-1 max-w-2xl text-sm text-ink-soft">{VERDICT[comparison.compatibility]?.body}</p>
              <ul className="mt-4 flex flex-wrap gap-x-6 gap-y-1 text-sm">
                {runIds.map((id, i) => {
                  const r = runInfo(id);
                  return (
                    <li key={id}>
                      <span className="text-ink-soft">Run {letter(i)}:</span>{" "}
                      <Link to={`/results/${id}`} className="underline underline-offset-4">
                        {r ? formatTimestamp(r.started_at) : "run"}
                      </Link>{" "}
                      <span className="font-mono text-xs text-ink-soft">{/^[0-9a-f]{16,}$/i.test(id) ? id.slice(0, 8) : id}</span>
                    </li>
                  );
                })}
              </ul>
            </section>

            <section className="rounded-xl border border-line p-6">
              <h3 className="font-semibold">What changed</h3>
              {(() => {
                const changed = changedSettings(comparison.configuration_diffs, runIds);
                if (changed.length === 0) {
                  return <p className="mt-2 text-sm text-ink-soft">Nothing. Both runs used exactly the same settings.</p>;
                }
                return (
                  <ul className="mt-3 divide-y divide-line text-sm">
                    {changed.map((c) => (
                      <li key={c.path} className="flex flex-col gap-1 py-2 sm:flex-row sm:justify-between sm:gap-6">
                        <span className="font-mono text-ink-soft break-all">{c.path}</span>
                        <span className="font-mono break-all">{c.values.join("  →  ")}</span>
                      </li>
                    ))}
                  </ul>
                );
              })()}
              {comparison.compatibility_reasons.length > 0 && (
                <details className="mt-4 text-sm">
                  <summary className="cursor-pointer text-ink-soft hover:text-ink">How this was decided</summary>
                  <ul className="mt-2 list-disc space-y-1 pl-5 text-ink-soft">
                    {comparison.compatibility_reasons.map((r, i) => (
                      <li key={i} className="break-words">{r}</li>
                    ))}
                  </ul>
                </details>
              )}
            </section>

            <section className="rounded-xl border border-line p-6">
              <h3 className="font-semibold">Results side by side</h3>
              <div className="mt-3 overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-line text-left text-ink-soft">
                      <th className="py-2 pr-4 font-normal">Measure</th>
                      {runIds.map((id, i) => (
                        <th key={id} className="py-2 pr-4 text-right font-normal">
                          Run {letter(i)}
                        </th>
                      ))}
                      {runIds.length === 2 && <th className="py-2 text-right font-normal">Change</th>}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-line">
                    {rows.map((row) => {
                      const [a, b] = row.values;
                      const delta = a != null && b != null ? b - a : null;
                      return (
                        <tr key={row.label}>
                          <td className="py-2 pr-4">{row.label}</td>
                          {row.values.map((v, i) => (
                            <td key={i} className="py-2 pr-4 text-right tabular-nums">
                              {row.format(v)}
                            </td>
                          ))}
                          {runIds.length === 2 && (
                            <td className="py-2 text-right tabular-nums text-ink-soft">
                              {a == null || b == null
                                ? "—"
                                : delta == null || Math.abs(delta) < 1e-9
                                ? "same"
                                : `${delta > 0 ? "+" : "−"}${row.format(Math.abs(delta))}`}
                            </td>
                          )}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </section>
          </div>
        )}
      </DataStateView>
    </div>
  );
}
