import { useState } from "react";
import { Link } from "react-router-dom";
import { useConnection } from "../context/ConnectionContext";
import { useComparison, useRuns } from "../data/useData";
import { DataStateView } from "../components/DataStateView";
import { EmptyState } from "../components/EmptyState";
import { evaluatorLabel, formatTimestamp } from "../lib/derived";
import type { ComparisonCompatibility } from "../types/domain";

// Demo-data examples. "pick" lets you compare any two runs (live API only:
// the bundled fixtures contain just these three precomputed comparisons).
const PRESETS = [
  { id: "pick", label: "Pick two runs" },
  { id: "identical", label: "Demo: same setup twice" },
  { id: "intervention", label: "Demo: 1 vs 2 passages" },
  { id: "incompatible", label: "Demo: RAG vs no RAG" },
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
  mrr: "How high the right passage ranked (MRR)",
};

function metricLabel(metric: string): string {
  if (METRIC_LABEL[metric]) return METRIC_LABEL[metric];
  const m = /^(hit|recall|precision)_at_k_k(\d+)$/.exec(metric);
  if (!m) return metric;
  if (m[1] === "hit") return `Right passage found in top ${m[2]}`;
  if (m[1] === "recall") return `Share of needed passages found (top ${m[2]})`;
  return `Share of top ${m[2]} passages that are relevant`;
}

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
  const { dataMode } = useConnection();
  const live = dataMode === "live";
  const [chosenPreset, setPreset] = useState(live ? "pick" : "identical");
  // "Pick two runs" needs the live API; in demo mode fall back to a demo preset.
  const preset = !live && chosenPreset === "pick" ? "identical" : chosenPreset;
  const sortedRuns = [...(allRuns ?? [])].sort((a, b) => b.started_at.localeCompare(a.started_at));
  const [pickA, setPickA] = useState("");
  const [pickB, setPickB] = useState("");
  const picked = preset === "pick" && pickA && pickB && pickA !== pickB ? [pickA, pickB] : undefined;
  const { data: comparison, loading, error, refetch } = useComparison(preset, picked);
  const waitingForPick = preset === "pick" && !picked;

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
          label: metricLabel(metric),
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
        {PRESETS.filter((p) => live || p.id !== "pick").map((p) => (
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

      {preset === "pick" && (
        <div className="mb-6 grid gap-3 sm:grid-cols-2">
          {[
            { label: "Run A", value: pickA, set: setPickA },
            { label: "Run B", value: pickB, set: setPickB },
          ].map(({ label, value, set }) => (
            <label key={label} className="flex min-w-0 flex-col gap-1 text-sm">
              <span className="text-ink-soft">{label}</span>
              <select
                value={value}
                onChange={(e) => set(e.target.value)}
                className="w-full min-w-0 rounded-md border border-line bg-paper px-3 py-2 text-sm"
              >
                <option value="">Choose a run…</option>
                {sortedRuns.map((r) => (
                  <option key={r.run_id} value={r.run_id}>
                    {r.experiment_id} · {r.adapter_id ?? "adapter not recorded"}
                    {r.repetition_index ? ` · repeat ${r.repetition_index}` : ""} · {formatTimestamp(r.started_at)}
                  </option>
                ))}
              </select>
            </label>
          ))}
        </div>
      )}

      {waitingForPick ? (
        <EmptyState
          title="Choose two different runs"
          detail="The comparison checks first whether the runs can be fairly compared (same questions, what changed), then shows the results side by side."
        />
      ) : (
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
                                : row.format === pct
                                ? // A difference of two rates is in percentage points, not percent.
                                  `${delta > 0 ? "+" : "−"}${Math.round(Math.abs(delta) * 100)} pts`
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
      )}
    </div>
  );
}
