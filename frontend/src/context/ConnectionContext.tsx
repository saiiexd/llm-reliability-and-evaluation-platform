/**
 * React Context managing backend API connectivity, data mode (Live API vs Fixture Demo),
 * and interactive execution actions.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import {
  ApiClient,
  defaultApiClient,
  type HealthResponse,
  type RunExecutionResponse,
} from "../api/client";
import type {
  BenchmarkExecutionRequest,
  BenchmarkExecutionResponse,
} from "../types/domain";

export type DataMode = "live" | "fixture";
export type ConnectionStatus = "connected" | "connecting" | "disconnected" | "error";

interface ConnectionContextValue {
  dataMode: DataMode;
  connectionStatus: ConnectionStatus;
  health: HealthResponse | null;
  apiBaseUrl: string;
  error: string | null;
  client: ApiClient;
  setDataMode: (mode: DataMode) => void;
  checkConnection: () => Promise<boolean>;
  executeRun: (
    experimentId: string,
    adapterId: string,
    allowExternalInference?: boolean,
    config?: Record<string, unknown>,
  ) => Promise<RunExecutionResponse>;
  executeBenchmark: (
    req: BenchmarkExecutionRequest,
  ) => Promise<BenchmarkExecutionResponse>;
}

const ConnectionContext = createContext<ConnectionContextValue | null>(null);

const STORAGE_KEY = "llm_reliability_data_mode";

export function ConnectionProvider({
  children,
  client = defaultApiClient,
}: {
  children: ReactNode;
  client?: ApiClient;
}) {
  // Default to 'live' if not saved, but if live fails during startup,
  // user will be warned and can switch to fixture mode.
  const [dataMode, setDataModeState] = useState<DataMode>(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    return saved === "fixture" ? "fixture" : "live";
  });

  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>("connecting");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const checkConnection = useCallback(async (): Promise<boolean> => {
    setConnectionStatus("connecting");
    setError(null);
    try {
      const data = await client.checkHealth();
      setHealth(data);
      setConnectionStatus("connected");
      return true;
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      setConnectionStatus("disconnected");
      return false;
    }
  }, [client]);

  const setDataMode = useCallback((mode: DataMode) => {
    setDataModeState(mode);
    localStorage.setItem(STORAGE_KEY, mode);
  }, []);

  const executeRun = useCallback(
    async (
      experimentId: string,
      adapterId: string,
      allowExternalInference: boolean = false,
      config?: Record<string, unknown>,
    ): Promise<RunExecutionResponse> => {
      if (dataMode !== "live") {
        throw new Error(
          "Run execution is not permitted in Fixture Mode. Switch to Live API mode to execute experiments.",
        );
      }
      return client.executeExperiment(experimentId, {
        adapter_id: adapterId,
        allow_external_inference: allowExternalInference,
        model_config: config,
      });
    },
    [client, dataMode],
  );

  const executeBenchmark = useCallback(
    async (req: BenchmarkExecutionRequest): Promise<BenchmarkExecutionResponse> => {
      if (dataMode !== "live") {
        throw new Error(
          "Benchmark execution is not permitted in Fixture Mode. Switch to Live API mode to execute benchmarks.",
        );
      }
      return client.executeBenchmark(req);
    },
    [client, dataMode],
  );

  useEffect(() => {
    void checkConnection();
  }, [checkConnection]);

  const value: ConnectionContextValue = {
    dataMode,
    connectionStatus,
    health,
    apiBaseUrl: client.getBaseUrl(),
    error,
    client,
    setDataMode,
    checkConnection,
    executeRun,
    executeBenchmark,
  };

  return <ConnectionContext.Provider value={value}>{children}</ConnectionContext.Provider>;
}

export function useConnection(): ConnectionContextValue {
  const ctx = useContext(ConnectionContext);
  if (!ctx) {
    throw new Error("useConnection must be used within a ConnectionProvider");
  }
  return ctx;
}
