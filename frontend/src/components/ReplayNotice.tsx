/**
 * Marks a run that re-evaluated the recorded answers of another run
 * (ReplayAdapter) instead of generating new ones. Its answers, latency and
 * token counts belong to the source run; only the checks are new.
 */
import { Link } from "react-router-dom";
import type { ExperimentRun } from "../types/domain";

export function replaySources(run: ExperimentRun): string[] {
  const ids = new Set<string>();
  for (const tc of run.result?.test_case_results ?? []) {
    const replay = tc.model_response?.provider_metadata?.replay as { source_run_id?: string } | undefined;
    if (replay?.source_run_id) ids.add(replay.source_run_id);
  }
  return [...ids];
}

export function ReplayNotice({ run }: { run: ExperimentRun }) {
  const sources = replaySources(run);
  if (sources.length === 0) return null;
  return (
    <p className="mt-4 rounded-md border border-dashed border-ink-soft/60 px-3 py-2 text-sm text-ink-soft">
      <span className="font-medium text-ink">Re-judged run</span> · the answers were recorded by{" "}
      {sources.map((id, i) => (
        <span key={id}>
          {i > 0 && ", "}
          <Link to={`/results/${id}`} className="font-mono text-xs underline underline-offset-4">
            {id.slice(0, 8)}
          </Link>
        </span>
      ))}{" "}
      and checked again here. Speed and token figures are from that original run.
    </p>
  );
}
