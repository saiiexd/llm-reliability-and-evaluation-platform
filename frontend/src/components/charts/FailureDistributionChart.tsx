import { formatCategoryLabel } from "../../lib/derived";

interface FailureDistributionChartProps {
  categoryCounts: Map<string, number> | Record<string, number>;
  totalFailures?: number;
  title?: string;
  emptyMessage?: string;
}

const CATEGORY_EXPLANATIONS: Record<string, string> = {
  incorrect: "Output did not match reference answer under evaluator criteria.",
  retrieval_failure: "Required reference context was not retrieved in top passages.",
  generation_failure: "Relevant context was retrieved, but model hallucinated or erred.",
  unsupported: "Answer contains claims not substantiated by retrieved context.",
  contradicted: "Answer directly contradicts facts in retrieved passages.",
  evaluator_disagreement: "Multiple evaluators split on whether the answer passed.",
  insufficient_evidence: "Evaluator skipped or lacked ground-truth evidence.",
  undetermined: "Available evidence does not isolate a single failure cause.",
};

export function FailureDistributionChart({
  categoryCounts,
  totalFailures,
  title = "Failure Mode Distribution",
  emptyMessage = "No failures detected across evaluated test cases (100% pass rate).",
}: FailureDistributionChartProps) {
  // Normalize input to array of [category, count]
  const entries: [string, number][] =
    categoryCounts instanceof Map
      ? Array.from(categoryCounts.entries())
      : Object.entries(categoryCounts);

  // Filter out zero counts and 'correct'
  const activeFailures = entries.filter(
    ([cat, count]) => count > 0 && cat.toLowerCase() !== "correct"
  );

  const sumFailures =
    totalFailures ?? activeFailures.reduce((acc, [, count]) => acc + count, 0);

  // Find max for scaling
  const maxCount = activeFailures.length > 0 ? Math.max(...activeFailures.map(([, c]) => c)) : 1;

  if (activeFailures.length === 0) {
    return (
      <div className="border border-line bg-paper p-5 flex flex-col items-center justify-center text-center h-full min-h-[160px]">
        <div className="w-8 h-8 rounded-full bg-emerald-950/60 border border-emerald-500/30 flex items-center justify-center text-emerald-400 mb-2">
          ✓
        </div>
        <h4 className="font-serif text-sm font-medium text-ink">{title}</h4>
        <p className="mt-1 text-xs text-ink-soft max-w-sm">{emptyMessage}</p>
      </div>
    );
  }

  return (
    <div
      className="border border-line bg-paper p-5"
      role="region"
      aria-label={`${title}: ${activeFailures.length} failure modes across ${sumFailures} failing test cases.`}
    >
      <div className="flex items-center justify-between border-b border-line pb-2 mb-3">
        <div>
          <h4 className="font-serif text-sm font-medium text-ink">{title}</h4>
          <span className="text-[11px] text-ink-soft">
            Breakdown of root causes identified by diagnostic taxonomy
          </span>
        </div>
        <span className="font-mono text-xs text-ink-soft">
          {sumFailures} {sumFailures === 1 ? "failure" : "failures"} total
        </span>
      </div>

      <div className="space-y-3.5">
        {activeFailures.map(([cat, count]) => {
          const ratio = sumFailures > 0 ? count / sumFailures : 0;
          const percentage = (ratio * 100).toFixed(1);
          const barWidthPercent = Math.max(8, (count / maxCount) * 100);
          const explanation = CATEGORY_EXPLANATIONS[cat.toLowerCase()] ?? "Evaluator failure mode.";

          return (
            <div key={cat} className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span className="font-medium text-ink flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-red-400 inline-block" aria-hidden="true" />
                  {formatCategoryLabel(cat)}
                </span>
                <span className="font-mono text-xs text-ink font-semibold">
                  {count} <span className="font-normal text-ink-soft">({percentage}%)</span>
                </span>
              </div>

              {/* Proportional Bar */}
              <div className="w-full bg-paper-dim border border-line h-4 relative overflow-hidden">
                <div
                  className="h-full bg-red-900/60 border-r-2 border-r-red-400 transition-all duration-500"
                  style={{ width: `${barWidthPercent}%` }}
                />
              </div>

              {/* Explanatory subtitle */}
              <p className="text-[11px] text-ink-soft italic leading-snug">
                {explanation}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
