import { formatPercent } from "../../lib/derived";

interface VisualMetricsBarProps {
  passed: number;
  failed: number;
  executionErrors?: number;
  total?: number;
  height?: number;
  showLabels?: boolean;
}

export function VisualMetricsBar({
  passed,
  failed,
  executionErrors = 0,
  total,
  height = 8,
  showLabels = true,
}: VisualMetricsBarProps) {
  const computedTotal = total ?? (passed + failed + executionErrors);
  const passRatio = computedTotal > 0 ? passed / computedTotal : 0;
  const failRatio = computedTotal > 0 ? failed / computedTotal : 0;
  const errorRatio = computedTotal > 0 ? executionErrors / computedTotal : 0;

  if (computedTotal === 0) {
    return (
      <div className="w-full bg-paper-dim border border-line h-2 rounded-full overflow-hidden opacity-40" />
    );
  }

  return (
    <div className="space-y-1.5 w-full">
      {/* Segmented bar */}
      <div
        className="w-full bg-paper-dim border border-line flex overflow-hidden rounded-xs"
        style={{ height }}
        role="progressbar"
        aria-valuenow={Math.round(passRatio * 100)}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`Pass rate: ${(passRatio * 100).toFixed(1)}%. ${passed} passed, ${failed} failed.`}
      >
        {passed > 0 && (
          <div
            className="bg-emerald-400 transition-all duration-500"
            style={{ width: `${passRatio * 100}%` }}
            title={`Passed: ${passed} (${(passRatio * 100).toFixed(1)}%)`}
          />
        )}
        {failed > 0 && (
          <div
            className="bg-red-400 transition-all duration-500"
            style={{ width: `${failRatio * 100}%` }}
            title={`Failed: ${failed} (${(failRatio * 100).toFixed(1)}%)`}
          />
        )}
        {executionErrors > 0 && (
          <div
            className="bg-amber-400 transition-all duration-500"
            style={{ width: `${errorRatio * 100}%` }}
            title={`Execution Errors: ${executionErrors} (${(errorRatio * 100).toFixed(1)}%)`}
          />
        )}
      </div>

      {showLabels && (
        <div className="flex items-center justify-between text-[11px] font-mono text-ink-soft">
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1 text-emerald-400">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
              {passed} pass ({formatPercent(passRatio, 0)})
            </span>
            <span className="flex items-center gap-1 text-red-400">
              <span className="w-1.5 h-1.5 rounded-full bg-red-400" />
              {failed} fail ({formatPercent(failRatio, 0)})
            </span>
            {executionErrors > 0 && (
              <span className="flex items-center gap-1 text-amber-400">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                {executionErrors} error
              </span>
            )}
          </div>
          <span>Total: {computedTotal}</span>
        </div>
      )}
    </div>
  );
}
