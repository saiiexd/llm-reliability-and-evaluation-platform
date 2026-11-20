/**
 * Loading / error wrapper used by every data-backed view.
 * Errors always offer a retry and an explicit switch to demo data.
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
      <div className="flex min-h-[200px] items-center justify-center gap-3 text-sm text-ink-soft">
        <div className="h-4 w-4 animate-spin rounded-full border-2 border-line border-t-ink" />
        Loading…
      </div>
    );
  }

  if (error) {
    const isApiError = error instanceof ApiError;
    return (
      <div className="rounded-xl border border-accent/40 bg-accent-soft/40 p-5 text-sm">
        <p className="font-semibold text-ink">Couldn't load this data</p>
        <p className="mt-1 text-ink-soft">{error.message}</p>
        {isApiError && error.requestId && (
          <p className="mt-1 font-mono text-xs text-ink-soft">
            Request {error.requestId} · HTTP {error.status}
          </p>
        )}
        <div className="mt-4 flex flex-wrap gap-2">
          {onRetry && (
            <button
              type="button"
              onClick={onRetry}
              className="rounded-md bg-ink px-3 py-1.5 text-xs font-medium text-paper hover:opacity-90"
            >
              Try again
            </button>
          )}
          <button
            type="button"
            onClick={() => setDataMode("fixture")}
            className="rounded-md border border-line px-3 py-1.5 text-xs text-ink hover:bg-paper-dim"
          >
            Use demo data
          </button>
        </div>
      </div>
    );
  }

  return <>{children}</>;
}
