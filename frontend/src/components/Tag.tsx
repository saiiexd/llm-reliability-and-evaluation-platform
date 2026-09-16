import type { ReactNode } from "react";
import type { Tone } from "../lib/tone";
import { TONE_CLASSES } from "../lib/tone";

/**
 * A small text label with restrained border/color treatment. Never the
 * sole carrier of information: always render alongside the underlying
 * evidence and explanation (see EvidencePanel), not as a standalone
 * verdict.
 */
export function Tag({ children, tone = "neutral" }: { children: ReactNode; tone?: Tone }) {
  return (
    <span
      className={`inline-block border px-2 py-0.5 font-mono text-xs whitespace-nowrap ${TONE_CLASSES[tone]}`}
    >
      {children}
    </span>
  );
}
