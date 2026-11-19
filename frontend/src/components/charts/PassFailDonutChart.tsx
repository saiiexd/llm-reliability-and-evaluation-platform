import { formatPercent } from "../../lib/derived";

interface PassFailDonutChartProps {
  passed: number;
  failed: number;
  executionErrors?: number;
  total?: number;
  size?: number;
  thickness?: number;
  title?: string;
  subtitle?: string;
}

export function PassFailDonutChart({
  passed,
  failed,
  executionErrors = 0,
  total,
  size = 180,
  thickness = 22,
  title,
  subtitle,
}: PassFailDonutChartProps) {
  const computedTotal = total ?? (passed + failed + executionErrors);
  const passRate = computedTotal > 0 ? (passed / computedTotal) * 100 : 0;

  // SVG Geometry
  const radius = (size - thickness) / 2;
  const circumference = 2 * Math.PI * radius;
  const center = size / 2;

  // Segment proportions
  const passRatio = computedTotal > 0 ? passed / computedTotal : 0;
  const failRatio = computedTotal > 0 ? failed / computedTotal : 0;
  const errorRatio = computedTotal > 0 ? executionErrors / computedTotal : 0;

  const passStroke = circumference * passRatio;
  const failStroke = circumference * failRatio;
  const errorStroke = circumference * errorRatio;

  // Rotational offsets (starting from top, 12 o'clock = -90deg)
  const passOffset = 0;
  const failOffset = -passStroke;
  const errorOffset = -(passStroke + failStroke);

  return (
    <div
      className="flex flex-col sm:flex-row items-center gap-6"
      role="img"
      aria-label={`Evaluation Pass Rate: ${passRate.toFixed(1)}%. ${passed} passed, ${failed} failed, ${executionErrors} execution errors of ${computedTotal} total.`}
    >
      {/* SVG Donut */}
      <div className="relative shrink-0" style={{ width: size, height: size }}>
        <svg
          width={size}
          height={size}
          viewBox={`0 0 ${size} ${size}`}
          className="transform -rotate-90"
        >
          {/* Background track */}
          <circle
            cx={center}
            cy={center}
            r={radius}
            fill="transparent"
            stroke="var(--color-line, #30363d)"
            strokeWidth={thickness}
            opacity={0.5}
          />

          {computedTotal === 0 ? (
            <circle
              cx={center}
              cy={center}
              r={radius}
              fill="transparent"
              stroke="var(--color-line, #30363d)"
              strokeWidth={thickness}
              strokeDasharray="4 4"
            />
          ) : (
            <>
              {/* Passed segment (Neutral Crisp Light / Emerald) */}
              {passed > 0 && (
                <circle
                  cx={center}
                  cy={center}
                  r={radius}
                  fill="transparent"
                  stroke="#34d399"
                  strokeWidth={thickness}
                  strokeDasharray={`${passStroke} ${circumference}`}
                  strokeDashoffset={passOffset}
                  strokeLinecap="round"
                  className="transition-all duration-500"
                />
              )}

              {/* Evaluation Failed segment (Coral / Negative) */}
              {failed > 0 && (
                <circle
                  cx={center}
                  cy={center}
                  r={radius}
                  fill="transparent"
                  stroke="#f87171"
                  strokeWidth={thickness}
                  strokeDasharray={`${failStroke} ${circumference}`}
                  strokeDashoffset={failOffset}
                  strokeLinecap={passed > 0 ? "butt" : "round"}
                  className="transition-all duration-500"
                />
              )}

              {/* Execution Error segment (Amber / Caution) */}
              {executionErrors > 0 && (
                <circle
                  cx={center}
                  cy={center}
                  r={radius}
                  fill="transparent"
                  stroke="#fbbf24"
                  strokeWidth={thickness}
                  strokeDasharray={`${errorStroke} ${circumference}`}
                  strokeDashoffset={errorOffset}
                  strokeLinecap="butt"
                  className="transition-all duration-500"
                />
              )}
            </>
          )}
        </svg>

        {/* Center Text: Big Pass Rate */}
        <div className="absolute inset-0 flex flex-col items-center justify-center text-center select-none pointer-events-none">
          <span className="font-serif text-3xl font-bold tracking-tight text-ink">
            {computedTotal > 0 ? `${passRate.toFixed(1)}%` : "N/A"}
          </span>
          <span className="font-mono text-[10px] uppercase tracking-wider text-ink-soft">
            Pass Rate
          </span>
        </div>
      </div>

      {/* Accessible Structured Legend */}
      <div className="flex flex-col justify-center gap-2 text-xs w-full max-w-xs">
        {title && (
          <h4 className="font-serif font-medium text-ink text-sm mb-1">{title}</h4>
        )}

        <div className="flex items-center justify-between border-b border-line pb-1.5">
          <div className="flex items-center gap-2">
            <span
              className="inline-block w-3 h-3 rounded-full bg-emerald-400"
              aria-hidden="true"
            />
            <span className="font-mono text-ink">Evaluation Passed</span>
          </div>
          <span className="font-mono font-semibold text-ink">
            {passed} <span className="font-normal text-ink-soft">({formatPercent(passRatio, 1)})</span>
          </span>
        </div>

        <div className="flex items-center justify-between border-b border-line pb-1.5">
          <div className="flex items-center gap-2">
            <span
              className="inline-block w-3 h-3 rounded-full bg-red-400"
              aria-hidden="true"
            />
            <span className="font-mono text-ink">Evaluation Failed</span>
          </div>
          <span className="font-mono font-semibold text-ink">
            {failed} <span className="font-normal text-ink-soft">({formatPercent(failRatio, 1)})</span>
          </span>
        </div>

        {executionErrors > 0 && (
          <div className="flex items-center justify-between border-b border-line pb-1.5">
            <div className="flex items-center gap-2">
              <span
                className="inline-block w-3 h-3 rounded-full bg-amber-400"
                aria-hidden="true"
              />
              <span className="font-mono text-ink">Execution Errors</span>
            </div>
            <span className="font-mono font-semibold text-amber-400">
              {executionErrors} <span className="font-normal text-ink-soft">({formatPercent(errorRatio, 1)})</span>
            </span>
          </div>
        )}

        <div className="flex items-center justify-between pt-0.5 text-ink-soft font-mono text-[11px]">
          <span>Total Test Cases</span>
          <span>{computedTotal}</span>
        </div>

        {subtitle && (
          <p className="mt-1 text-[11px] text-ink-soft leading-normal italic">
            {subtitle}
          </p>
        )}
      </div>
    </div>
  );
}
