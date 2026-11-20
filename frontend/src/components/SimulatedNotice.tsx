/**
 * Marks a run whose answers or checks were simulated (mock adapter, canned
 * judge, fake embeddings). The list comes from the backend
 * (`simulated_components`), so the frontend never guesses from names.
 */
import type { ExperimentRun } from "../types/domain";
import { Tag } from "./Tag";

export function isSimulated(run: Pick<ExperimentRun, "simulated_components">): boolean {
  return (run.simulated_components?.length ?? 0) > 0;
}

export function SimulatedTag({ run }: { run: Pick<ExperimentRun, "simulated_components"> }) {
  if (!isSimulated(run)) return null;
  return <Tag>Simulated</Tag>;
}

export function SimulatedNotice({ run }: { run: Pick<ExperimentRun, "simulated_components"> }) {
  if (!isSimulated(run)) return null;
  return (
    <details className="mt-4 rounded-md border border-dashed border-ink-soft/60 px-3 py-2 text-sm">
      <summary className="cursor-pointer text-ink-soft">
        <span className="font-medium text-ink">Simulated run</span> · these numbers test the
        platform, not a real model. Show what was simulated
      </summary>
      <ul className="mt-2 list-disc space-y-1 pl-5 text-xs text-ink-soft">
        {run.simulated_components!.map((c) => (
          <li key={c}>{c}</li>
        ))}
      </ul>
    </details>
  );
}
