// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import App from "../App";
import { ConnectionProvider } from "../context/ConnectionContext";
import { ApiClient } from "../api/client";

describe("Simplified Academic Research Navigation & Layout", () => {
  let mockClient: ApiClient;

  beforeEach(() => {
    localStorage.clear();
    mockClient = new ApiClient("http://localhost:8000/api/v1");
    vi.spyOn(mockClient, "checkHealth").mockResolvedValue({
      status: "healthy",
      version: "0.1.0",
      environment: "test",
      timestamp: "2026-11-02T00:00:00Z",
      database: "local_filesystem",
      allowlisted_adapters: ["mock-adapter-v1"],
    });
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("renders exactly the four primary academic navigation sections in the header", () => {
    render(
      <ConnectionProvider client={mockClient}>
        <MemoryRouter initialEntries={["/"]}>
          <App />
        </MemoryRouter>
      </ConnectionProvider>,
    );

    // Header branding
    expect(screen.getByText("LLM Reliability")).toBeDefined();
    expect(screen.getByText("Research Platform")).toBeDefined();

    // The four core navigation items
    const overviewLinks = screen.getAllByRole("link", { name: "Overview" });
    expect(overviewLinks.length).toBeGreaterThan(0);

    const experimentsLinks = screen.getAllByRole("link", { name: "Experiments" });
    expect(experimentsLinks.length).toBeGreaterThan(0);

    const resultsLinks = screen.getAllByRole("link", { name: "Results" });
    expect(resultsLinks.length).toBeGreaterThan(0);

    const analysisLinks = screen.getAllByRole("link", { name: "Analysis" });
    expect(analysisLinks.length).toBeGreaterThan(0);
  });

  it("renders academic introduction and clear primary action on the Overview page", () => {
    render(
      <ConnectionProvider client={mockClient}>
        <MemoryRouter initialEntries={["/"]}>
          <App />
        </MemoryRouter>
      </ConnectionProvider>,
    );

    // One plain question as the headline, plus a one-sentence explanation
    expect(screen.getByRole("heading", { name: /Is the model answering correctly\?/i })).toBeDefined();
    expect(screen.getByText(/checks every answer automatically/i)).toBeDefined();

    // Primary action button
    const startBtn = screen.getByRole("link", { name: /Start an Experiment/i });
    expect(startBtn).toBeDefined();
    expect(startBtn.getAttribute("href")).toContain("/experiments");

    // Three plain steps explaining the workflow
    expect(screen.getByText("Define an experiment")).toBeDefined();
    expect(screen.getByText("Run it")).toBeDefined();
    expect(screen.getByText("Read the results")).toBeDefined();
  });

  it("renders the Experiments section with guided definition trigger", () => {
    render(
      <ConnectionProvider client={mockClient}>
        <MemoryRouter initialEntries={["/experiments"]}>
          <App />
        </MemoryRouter>
      </ConnectionProvider>,
    );

    expect(screen.getByRole("heading", { name: "Experiments" })).toBeDefined();
    expect(screen.getByRole("button", { name: /\+ Define.*Experiment/ })).toBeDefined();
  });

  it("renders the Analysis section with the 5 plain-language diagnostic sub-tabs", () => {
    render(
      <ConnectionProvider client={mockClient}>
        <MemoryRouter initialEntries={["/analysis"]}>
          <App />
        </MemoryRouter>
      </ConnectionProvider>,
    );

    expect(screen.getByText("Analytical Diagnostics")).toBeDefined();
    expect(screen.getByRole("button", { name: "Model Comparison" })).toBeDefined();
    expect(screen.getByRole("button", { name: "Reliability & Failure Forensics" })).toBeDefined();
    expect(screen.getByRole("button", { name: "RAG & Context Retrieval" })).toBeDefined();
    expect(screen.getByRole("button", { name: "Consistency & Robustness" })).toBeDefined();
    expect(screen.getByRole("button", { name: "System Latency & Telemetry" })).toBeDefined();
  });

  it("seamlessly redirects legacy comparisons route to the Analysis Model Comparison tab", () => {
    render(
      <ConnectionProvider client={mockClient}>
        <MemoryRouter initialEntries={["/comparisons"]}>
          <App />
        </MemoryRouter>
      </ConnectionProvider>,
    );

    // Successfully renders the Analysis view
    expect(screen.getByText("Analytical Diagnostics")).toBeDefined();
    expect(screen.getByRole("button", { name: "Model Comparison" })).toBeDefined();
  });
});

