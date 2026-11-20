// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, fireEvent, waitFor } from "@testing-library/react";
import { ConnectionProvider, useConnection } from "../context/ConnectionContext";
import { ConnectionBanner, ConnectionIndicator } from "../components/ConnectionBanner";
import { FixtureNotice } from "../components/FixtureNotice";
import { ApiClient } from "../api/client";

function TestConsumer() {
  const { dataMode, connectionStatus, setDataMode } = useConnection();
  return (
    <div>
      <span data-testid="mode">{dataMode}</span>
      <span data-testid="status">{connectionStatus}</span>
      <button type="button" onClick={() => setDataMode("live")}>
        Set Live
      </button>
      <button type="button" onClick={() => setDataMode("fixture")}>
        Set Fixture
      </button>
    </div>
  );
}

describe("Connection and Data Mode Context", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("defaults to live mode and connects when health check succeeds", async () => {
    const mockClient = new ApiClient("http://localhost:8000/api/v1");
    vi.spyOn(mockClient, "checkHealth").mockResolvedValue({
      status: "healthy",
      version: "0.1.0",
      environment: "test",
      timestamp: "2026-11-02T00:00:00Z",
      database: "local_filesystem",
      allowlisted_adapters: ["mock-adapter-v1"],
    });

    render(
      <ConnectionProvider client={mockClient}>
        <TestConsumer />
        <ConnectionIndicator />
      </ConnectionProvider>,
    );

    expect(screen.getByTestId("mode").textContent).toBe("live");

    await waitFor(() => {
      expect(screen.getByTestId("status").textContent).toBe("connected");
    });

    expect(screen.getByText("Live API")).toBeDefined();
  });

  it("detects disconnected backend and displays offline indicator and strict banner", async () => {
    const mockClient = new ApiClient("http://localhost:8000/api/v1");
    vi.spyOn(mockClient, "checkHealth").mockRejectedValue(new Error("Connection refused"));

    render(
      <ConnectionProvider client={mockClient}>
        <TestConsumer />
        <ConnectionIndicator />
        <ConnectionBanner />
      </ConnectionProvider>,
    );

    await waitFor(() => {
      expect(screen.getByTestId("status").textContent).toBe("disconnected");
    });

    // Verify prominent warning banner informs user about strict fallback policy
    expect(screen.getByText("Live API Unreachable")).toBeDefined();
    expect(screen.getByText(/The console will not silently substitute fixture data/)).toBeDefined();
    expect(screen.getByText("Retry Connection")).toBeDefined();
    expect(screen.getByText("Switch to Demo Mode")).toBeDefined();
  });

  it("allows switching to fixture mode explicitly and displays FixtureNotice", async () => {
    const mockClient = new ApiClient("http://localhost:8000/api/v1");
    vi.spyOn(mockClient, "checkHealth").mockResolvedValue({
      status: "healthy",
      version: "0.1.0",
      environment: "test",
      timestamp: "2026-11-02T00:00:00Z",
      database: "local_filesystem",
      allowlisted_adapters: ["mock-adapter-v1"],
    });

    render(
      <ConnectionProvider client={mockClient}>
        <TestConsumer />
        <ConnectionIndicator />
        <FixtureNotice />
      </ConnectionProvider>,
    );

    // Click Set Fixture
    fireEvent.click(screen.getByText("Set Fixture"));

    expect(screen.getByTestId("mode").textContent).toBe("fixture");
    expect(localStorage.getItem("llm_reliability_data_mode")).toBe("fixture");

    // FixtureNotice now renders
    expect(screen.getByText("Fixture data mode")).toBeDefined();
  });
});
