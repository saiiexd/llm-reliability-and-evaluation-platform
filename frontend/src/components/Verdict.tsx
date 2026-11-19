/**
 * Shared visuals for the four test-case outcomes (see lib/derived.ts
 * `testCaseVerdict`): a tag for one case and a bar + legend for a run.
 */
import { VERDICT_LABEL, type Verdict } from "../lib/derived";
import type { Tone } from "../lib/tone";
import { Tag } from "./Tag";

const VERDICT_TONE: Record<Verdict, Tone> = {
  passed: "neutral",
  disputed: "muted",
  failed: "negative",
  unscored: "muted",
  error: "negative",
};

export const VERDICT_HELP: Record<Verdict, string> = {
  passed: "Every check ran and accepted the answer.",
  disputed:
    "Two checks of the same thing disagree (e.g. exact match says wrong, the LLM judge says correct). Open the question to see both.",
  failed: "At least one check judged the answer wrong.",
  unscored: "No check said it was wrong, but at least one check could not run.",
  error: "The model call failed, so there was no answer to check.",
};

export function VerdictTag({ verdict }: { verdict: Verdict }) {
  return <Tag tone={VERDICT_TONE[verdict]}>{VERDICT_LABEL[verdict]}</Tag>;
}

const SEGMENT_CLASS: Record<Verdict, string> = {
  passed: "bg-ink",
  disputed: "bg-ink/45",
  failed: "bg-accent",
  unscored:
    "bg-[repeating-linear-gradient(135deg,var(--color-ink-soft)_0_2px,transparent_2px_6px)] opacity-70",
  error: "bg-accent/50",
};

const ORDER: Verdict[] = ["passed", "disputed", "failed", "unscored", "error"];

export function VerdictBar({
  counts,
  showLegend = true,
}: {
  counts: Record<Verdict, number> & { total: number };
  showLegend?: boolean;
}) {
  const { total } = counts;
  if (total === 0) {
    return <div className="h-2 w-full rounded-full bg-paper-raised" />;
  }
  const visible = ORDER.filter((v) => counts[v] > 0);
  return (
    <div className="w-full">
      <div
        className="flex h-2 w-full overflow-hidden rounded-full bg-paper-raised"
        role="img"
        aria-label={visible.map((v) => `${counts[v]} ${VERDICT_LABEL[v].toLowerCase()}`).join(", ")}
      >
        {visible.map((v) => (
          <div
            key={v}
            className={SEGMENT_CLASS[v]}
            style={{ width: `${(counts[v] / total) * 100}%` }}
          />
        ))}
      </div>
      {showLegend && (
        <ul className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-xs text-ink-soft">
          {visible.map((v) => (
            <li key={v} className="flex items-center gap-1.5" title={VERDICT_HELP[v]}>
              <span className={`inline-block h-2 w-2 rounded-full ${SEGMENT_CLASS[v]}`} />
              <span className="tabular-nums text-ink">{counts[v]}</span> {VERDICT_LABEL[v].toLowerCase()}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** One plain sentence describing a run's outcome. */
export function verdictSentence(counts: Record<Verdict, number> & { total: number }): string {
  const { total, passed, disputed, failed, unscored, error } = counts;
  if (total === 0) return "This run has no test cases.";
  const parts = [`${passed} of ${total} answers passed`];
  if (disputed) parts.push(`the checks disagreed on ${disputed}`);
  if (failed) parts.push(`${failed} ${failed === 1 ? "was" : "were"} wrong`);
  if (unscored) parts.push(`${unscored} could not be fully scored`);
  if (error) parts.push(`${error} crashed`);
  if (parts.length === 1) return `${parts[0]}.`;
  return `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}.`;
}

/**
 * Passed vs. not passed, for places where only run summaries are available
 * (they don't say whether a non-pass was wrong or just unscored).
 */
export function PassBar({ passed, total }: { passed: number; total: number }) {
  const pct = total > 0 ? (passed / total) * 100 : 0;
  return (
    <div
      className="h-2 w-full overflow-hidden rounded-full bg-ink-soft/25"
      role="img"
      aria-label={`${passed} of ${total} passed`}
    >
      <div className="h-full bg-ink" style={{ width: `${pct}%` }} />
    </div>
  );
}
