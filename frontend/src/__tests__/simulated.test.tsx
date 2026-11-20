// @vitest-environment jsdom
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { SimulatedNotice, SimulatedTag, isSimulated } from "../components/SimulatedNotice";
import runs from "../data/fixtures/runs.json";

describe("simulated labelling", () => {
  it("is driven only by the backend's simulated_components list", () => {
    expect(isSimulated({ simulated_components: [] })).toBe(false);
    expect(isSimulated({})).toBe(false);
    expect(isSimulated({ simulated_components: ["generation: mock"] })).toBe(true);
  });

  it("renders nothing for a real run", () => {
    const { container } = render(<SimulatedTag run={{ simulated_components: [] }} />);
    expect(container.textContent).toBe("");
  });

  it("lists every simulated component", () => {
    render(<SimulatedNotice run={{ simulated_components: ["generation: mock", "judge: canned"] }} />);
    expect(screen.getByText("generation: mock")).toBeTruthy();
    expect(screen.getByText("judge: canned")).toBeTruthy();
  });

  it("every bundled fixture run is marked simulated", () => {
    for (const run of Object.values(runs) as { simulated_components?: string[] }[]) {
      expect(run.simulated_components?.length ?? 0).toBeGreaterThan(0);
    }
  });
});
