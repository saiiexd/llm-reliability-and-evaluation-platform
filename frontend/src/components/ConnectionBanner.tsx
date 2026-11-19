/**
 * Visual Connection Banner and Status Indicator.
 *
 * Clearly displays whether the console is operating in Live API Mode
 * or Offline Fixture Demo Mode. If live API communication fails,
 * surfaces a strict error banner rather than silently falling back
 * to fixture data, allowing the user to retry or switch explicitly.
 */

import { useState } from "react";
import { useConnection } from "../context/ConnectionContext";

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
      <div className="flex items-center gap-2 text-xs">
        <span className="flex items-center gap-1.5 rounded-full border border-amber-500/40 bg-amber-500/10 px-2.5 py-0.5 font-mono text-amber-400">
          <span className="h-1.5 w-1.5 rounded-full bg-amber-400" />
          Fixture Mode
        </span>
        <button
          type="button"
          onClick={() => setDataMode("live")}
          className="border border-line px-2 py-0.5 text-xs text-ink-soft hover:border-ink hover:text-ink transition-colors"
          title="Switch to live FastAPI backend"
        >
          Use Live API
        </button>
      </div>
    );
  }

  // Live Mode
  return (
    <div className="flex items-center gap-2 text-xs">
      {connectionStatus === "connected" && (
        <span className="flex items-center gap-1.5 rounded-full border border-emerald-500/40 bg-emerald-500/10 px-2.5 py-0.5 font-mono text-emerald-400">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
          Live API
        </span>
      )}
      {connectionStatus === "connecting" && (
        <span className="flex items-center gap-1.5 rounded-full border border-line bg-paper-dim px-2.5 py-0.5 font-mono text-ink-soft">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-ink-soft" />
          Connecting...
        </span>
      )}
      {(connectionStatus === "disconnected" || connectionStatus === "error") && (
        <span className="flex items-center gap-1.5 rounded-full border border-red-500/40 bg-red-500/10 px-2.5 py-0.5 font-mono text-red-400">
          <span className="h-1.5 w-1.5 rounded-full bg-red-400" />
          API Offline
        </span>
      )}

      {connectionStatus === "disconnected" && (
        <button
          type="button"
          onClick={handleRetry}
          disabled={retrying}
          className="border border-line px-2 py-0.5 text-xs text-ink-soft hover:border-ink hover:text-ink disabled:opacity-50 transition-colors"
        >
          {retrying ? "Retrying..." : "Retry"}
        </button>
      )}

      <button
        type="button"
        onClick={() => setDataMode("fixture")}
        className="border border-line px-2 py-0.5 text-xs text-ink-soft hover:border-ink hover:text-ink transition-colors"
        title="Switch to bundled sample fixtures"
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

  if (dataMode === "live" && (connectionStatus === "disconnected" || connectionStatus === "error")) {
    return (
      <div className="mb-6 border-l-4 border-red-500 bg-red-950/40 border-y border-r border-line p-4 text-xs text-red-200">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="font-semibold text-red-300">Live API Unreachable</p>
            <p className="mt-0.5 text-red-200/90">
              Cannot connect to backend endpoint at{" "}
              <code className="bg-red-900/50 px-1 py-0.5 font-mono text-red-200">{apiBaseUrl}</code>.
            </p>
            <p className="mt-1 text-red-200/80">
              <strong>Strict Fallback Policy:</strong> The console will not silently substitute
              fixture data when an API error occurs. Please ensure the backend server is running (
              <code className="bg-red-900/50 px-1 py-0.5 font-mono text-red-200">
                uvicorn llm_reliability.api.app:app --port 8000
              </code>
              ) or consciously switch to Fixture Demo Mode below.
            </p>
            {error && <p className="mt-1 font-mono text-[11px] text-red-400">{error}</p>}
          </div>
          <div className="mt-2 flex shrink-0 items-center gap-2 sm:mt-0">
            <button
              type="button"
              onClick={handleRetry}
              disabled={retrying}
              className="bg-red-800 px-3 py-1.5 font-medium text-white hover:bg-red-700 disabled:opacity-50 transition-colors"
            >
              {retrying ? "Checking..." : "Retry Connection"}
            </button>
            <button
              type="button"
              onClick={() => setDataMode("fixture")}
              className="border border-red-700/60 bg-red-900/30 px-3 py-1.5 text-red-200 hover:bg-red-900/50 transition-colors"
            >
              Switch to Demo Mode
            </button>
          </div>
        </div>
      </div>
    );
  }

  return null;
}

