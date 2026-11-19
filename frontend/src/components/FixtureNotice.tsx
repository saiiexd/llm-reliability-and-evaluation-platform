/**
 * Notice displayed when the application is operating in Fixture Demo Mode.
 *
 * When the user switches to Live API mode, this banner disappears, replaced
 * by the live connection indicator in the application header.
 */

import { useConnection } from "../context/ConnectionContext";
import { fixtureMeta } from "../data/dataAccess";

export function FixtureNotice() {
  const { dataMode, setDataMode } = useConnection();

  if (dataMode !== "fixture") {
    return null;
  }

  return (
    <div className="mb-6 flex flex-col items-start justify-between gap-2 border border-amber-500/40 bg-amber-950/30 px-3.5 py-2.5 text-xs text-amber-200 sm:flex-row sm:items-center">
      <div>
        <span className="font-mono uppercase tracking-wide font-medium text-amber-400">
          Fixture data mode
        </span>{" "}
        <span className="text-amber-200/90">{fixtureMeta.warning}</span>
      </div>
      <button
        type="button"
        onClick={() => setDataMode("live")}
        className="shrink-0 border border-amber-500/50 bg-amber-900/40 px-2.5 py-1 text-xs font-medium text-amber-200 hover:bg-amber-900/70 transition-colors"
      >
        Connect to Live API
      </button>
    </div>
  );
}
