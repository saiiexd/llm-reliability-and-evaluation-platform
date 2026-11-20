/**
 * Shown on every page while the console uses bundled sample data instead
 * of the live backend, so nobody mistakes demo numbers for real results.
 */

import { useConnection } from "../context/ConnectionContext";
import { fixtureMeta } from "../data/dataAccess";

export function FixtureNotice() {
  const { dataMode, setDataMode } = useConnection();

  if (dataMode !== "fixture") {
    return null;
  }

  return (
    <div className="mb-8 flex flex-col items-start justify-between gap-3 rounded-xl border border-dashed border-ink-soft/60 px-4 py-3 text-sm sm:flex-row sm:items-center">
      <p className="text-ink-soft">
        <span className="font-medium text-ink">Fixture data mode</span> · {fixtureMeta.warning}
      </p>
      <button
        type="button"
        onClick={() => setDataMode("live")}
        className="shrink-0 rounded-md border border-line px-3 py-1 text-xs text-ink hover:bg-paper-dim"
      >
        Connect to Live API
      </button>
    </div>
  );
}
