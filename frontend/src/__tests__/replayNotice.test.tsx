// @vitest-environment jsdom
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { ReplayNotice, replaySources } from "../components/ReplayNotice";
import type { ExperimentRun } from "../types/domain";

function run(meta: Record<string, unknown>[]): ExperimentRun {
  return {
    run_id: "r",
    result: {
      test_case_results: meta.map((m, i) => ({
        test_case_id: `t${i}`,
        model_response: { output_text: "a", provider_metadata: m },
      })),
    },
  } as unknown as ExperimentRun;
}

describe("ReplayNotice", () => {
  it("lists distinct source runs and links to them", () => {
    const r = run([{ replay: { source_run_id: "abcdef1234" } }, { replay: { source_run_id: "abcdef1234" } }]);
    expect(replaySources(r)).toEqual(["abcdef1234"]);
    render(
      <MemoryRouter>
        <ReplayNotice run={r} />
      </MemoryRouter>,
    );
    expect(screen.getByText(/Re-judged run/)).toBeDefined();
    expect(screen.getByRole("link", { name: "abcdef12" }).getAttribute("href")).toBe("/results/abcdef1234");
  });

  it("renders nothing for ordinary runs", () => {
    const { container } = render(
      <MemoryRouter>
        <ReplayNotice run={run([{ rag: {} }])} />
      </MemoryRouter>,
    );
    expect(container.textContent).toBe("");
  });
});
