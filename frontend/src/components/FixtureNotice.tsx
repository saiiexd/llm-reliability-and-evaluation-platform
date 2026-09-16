/**
 * Shown on every page that renders fixture data, so it is never mistaken
 * for a live experiment result. See `data/dataAccess.ts` and
 * `data/fixtures/meta.json` for what generated this data and why.
 */
import { fixtureMeta } from "../data/dataAccess";

export function FixtureNotice() {
  return (
    <div className="mb-6 border border-line bg-paper-dim px-3 py-2 text-xs text-ink-soft">
      <span className="font-mono uppercase tracking-wide text-ink">Fixture data</span>{" "}
      {fixtureMeta.warning}
    </div>
  );
}
