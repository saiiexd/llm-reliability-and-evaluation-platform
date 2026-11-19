/**
 * Data-source indicator (header) and the "backend unreachable" banner.
 *
 * Live mode never silently falls back to fixture data: if the API is down
 * the user sees why and chooses to retry or switch to demo data.
 */

import { useState } from "react";
import { useConnection } from "../context/ConnectionContext";

const pill = "flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs";
const quietButton =
  "rounded-md px-2 py-0.5 text-xs text-ink-soft hover:bg-paper-dim hover:text-ink transition-colors disabled:opacity-50";

export function ConnectionIndicator() {
  const { dataMode, connectionStatus, setDataMode, checkConnection } = useConnection();
  const [retrying, setRetrying] = useState(false);

  const handleRetry = async () => {
    setRetrying(true);
    await checkConnection();
    setRetrying(false);
  };

  if (dataMode === "fixture") {
    return (
      <div className="flex items-center gap-1 text-xs">
        <span className={`${pill} border-dashed border-ink-soft text-ink-soft`}>Demo data</span>
        <button type="button" onClick={() => setDataMode("live")} className={quietButton}>
          Use Live API
        </button>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-1 text-xs">
      {connectionStatus === "connected" && (
        <span className={`${pill} border-line text-ink`}>
          <span className="h-1.5 w-1.5 rounded-full bg-ink" />
          Live API
        </span>
      )}
      {connectionStatus === "connecting" && (
        <span className={`${pill} border-line text-ink-soft`}>
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-ink-soft" />
          Connecting
        </span>
      )}
      {(connectionStatus === "disconnected" || connectionStatus === "error") && (
        <span className={`${pill} border-accent/40 text-accent`}>
          <span className="h-1.5 w-1.5 rounded-full bg-accent" />
          API offline
        </span>
      )}
      {connectionStatus === "disconnected" && (
        <button type="button" onClick={handleRetry} disabled={retrying} className={quietButton}>
          {retrying ? "Retrying" : "Retry"}
        </button>
      )}
      <button
        type="button"
        onClick={() => setDataMode("fixture")}
        className={quietButton}
        title="Show bundled sample data instead of the live backend"
      >
        Demo Mode
      </button>
    </div>
  );
}

export function ConnectionBanner() {
  const { dataMode, connectionStatus, apiBaseUrl, error, setDataMode, checkConnection } =
    useConnection();
  const [retrying, setRetrying] = useState(false);

  const handleRetry = async () => {
    setRetrying(true);
    await checkConnection();
    setRetrying(false);
  };

  if (dataMode !== "live" || (connectionStatus !== "disconnected" && connectionStatus !== "error")) {
    return null;
  }

  return (
    <div className="mb-8 rounded-xl border border-accent/40 bg-accent-soft/40 p-5 text-sm">
      <p className="font-semibold text-ink">Live API Unreachable</p>
      <p className="mt-1 text-ink-soft">
        Can't reach the backend at <code className="font-mono text-ink">{apiBaseUrl}</code>. Start
        it with <code className="font-mono text-ink">python run.py</code> from the project root.
      </p>
      <p className="mt-1 text-ink-soft">
        The console will not silently substitute fixture data, so nothing below is real until it
        reconnects.
      </p>
      {error && <p className="mt-2 font-mono text-xs text-accent">{error}</p>}
      <div className="mt-4 flex flex-wrap gap-2">
        <button
          type="button"
          onClick={handleRetry}
          disabled={retrying}
          className="rounded-md bg-ink px-3 py-1.5 text-xs font-medium text-paper hover:opacity-90 disabled:opacity-50"
        >
          {retrying ? "Checking" : "Retry Connection"}
        </button>
        <button
          type="button"
          onClick={() => setDataMode("fixture")}
          className="rounded-md border border-line px-3 py-1.5 text-xs text-ink hover:bg-paper-dim"
        >
          Switch to Demo Mode
        </button>
      </div>
    </div>
  );
}
