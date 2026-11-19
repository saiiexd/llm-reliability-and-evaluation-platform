/**
 * Reusable loading, error, and empty state container for console views.
 *
 * Ensures errors in live API communication clearly provide retry actions
 * and mode switches, while loading spinners match the editorial design.
 */

import { useConnection } from "../context/ConnectionContext";
import { ApiError } from "../api/client";

interface DataStateViewProps {
  loading: boolean;
  error: Error | null;
  onRetry?: () => void;
  children: React.ReactNode;
}

export function DataStateView({ loading, error, onRetry, children }: DataStateViewProps) {
  const { setDataMode } = useConnection();

  if (loading) {
    return (
      <div className="flex min-h-[240px] flex-col items-center justify-center border border-line bg-paper p-8 text-center">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-line border-t-ink" />
        <p className="mt-3 font-mono text-xs uppercase tracking-wider text-ink-soft">
          Loading live research data...
        </p>
      </div>
    );
  }

  if (error) {
    const isApiError = error instanceof ApiError;
    return (
      <div className="border border-red-500/40 bg-red-950/30 p-6 text-sm text-red-200">
        <h3 className="font-serif text-base font-semibold text-red-300">
          Failed to load research data
        </h3>
        <p className="mt-1 text-xs text-red-200/90">{error.message}</p>

        {isApiError && error.requestId && (
          <p className="mt-1 font-mono text-[11px] text-red-400">
            Request ID: {error.requestId} | Status: {error.status}
          </p>
        )}

        <div className="mt-4 flex flex-wrap items-center gap-3">
          {onRetry && (
            <button
              type="button"
              onClick={onRetry}
              className="bg-red-800 px-3 py-1.5 text-xs font-medium text-white hover:bg-red-700 transition-colors"
            >
              Retry Request
            </button>
          )}
          <button
            type="button"
            onClick={() => setDataMode("fixture")}
            className="border border-red-700/60 bg-red-900/30 px-3 py-1.5 text-xs text-red-200 hover:bg-red-900/50 transition-colors"
          >
            Switch to Demo Fixture Mode
          </button>
        </div>
      </div>
    );
  }

  return <>{children}</>;
}

