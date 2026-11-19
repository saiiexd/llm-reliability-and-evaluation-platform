/**
 * "Do the checks agree?" for one run.
 *
 * Two checks that judge the same thing (e.g. exact match and an LLM judge,
 * both judging correctness) can disagree. Hiding that would make either
 * one look like ground truth, so each pair is shown with raw agreement,
 * Cohen's kappa (agreement beyond chance) and the 2x2 counts. Pairs that
 * judge different things (correctness vs faithfulness) are never compared;
 * the backend already excludes them.
 */
import { useEvaluatorAnalytics } from "../data/useData";
import { evaluatorLabel } from "../lib/derived";
import type { EvaluatorAgreementSummary } from "../types/domain";

function kappaWords(k: number | null | undefined): string {
  if (k === null || k === undefined) return "not defined (one check gave the same verdict every time)";
  // Landis & Koch (1977) bands, used only as a rough reading aid.
  if (k < 0) return "worse than chance";
  if (k < 0.21) return "slight";
  if (k < 0.41) return "fair";
  if (k < 0.61) return "moderate";
  if (k < 0.81) return "substantial";
  return "almost perfect";
}

function Pair({ a }: { a: EvaluatorAgreementSummary }) {
  const nameA = evaluatorLabel(a.evaluator_a);
  const nameB = evaluatorLabel(a.evaluator_b);
  const rate = a.agreement_rate === null ? null : Math.round(a.agreement_rate * 100);
  return (
    <li className="py-3">
      <p className="text-sm">
        <span className="font-medium">{nameA}</span> vs <span className="font-medium">{nameB}</span>
      </p>
      <p className="mt-1 text-sm text-ink-soft">
        Same verdict on {a.agreements} of {a.cases_compared} questions
        {rate !== null && ` (${rate}%)`}. Beyond chance:{" "}
        {a.cohens_kappa === null || a.cohens_kappa === undefined ? "" : `κ = ${a.cohens_kappa.toFixed(2)}, `}
        {kappaWords(a.cohens_kappa)}.
      </p>
      {a.both_passed !== undefined && (
        <p className="mt-1 font-mono text-xs text-ink-soft">
          both pass {a.both_passed} · only {nameA} {a.only_a_passed} · only {nameB} {a.only_b_passed} · both
          fail {a.both_failed}
          {a.excluded_cases > 0 && ` · ${a.excluded_cases} not comparable`}
        </p>
      )}
    </li>
  );
}

export function EvaluatorAgreement({ runId }: { runId: string }) {
  const { data } = useEvaluatorAnalytics(runId);
  const pairs = (data?.agreements ?? []).filter((a) => a.cases_compared > 0);
  if (pairs.length === 0) return null;
  return (
    <details className="rounded-xl border border-line px-6 py-4">
      <summary className="cursor-pointer text-sm font-medium">
        Do the checks agree?{" "}
        <span className="font-normal text-ink-soft">
          ({pairs.length} pair{pairs.length > 1 ? "s" : ""} of checks that judge the same thing)
        </span>
      </summary>
      <ul className="mt-2 divide-y divide-line">
        {pairs.map((a) => (
          <Pair key={`${a.evaluator_a}-${a.evaluator_b}`} a={a} />
        ))}
      </ul>
    </details>
  );
}
