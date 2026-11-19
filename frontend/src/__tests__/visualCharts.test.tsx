// @vitest-environment jsdom
import { afterEach, describe, expect, it } from "vitest";
import { CATEGORY_HELP } from "../lib/derived";
import { cleanup, render, screen } from "@testing-library/react";
import { PassFailDonutChart } from "../components/charts/PassFailDonutChart";
import { FailureDistributionChart } from "../components/charts/FailureDistributionChart";
import { VisualMetricsBar } from "../components/charts/VisualMetricsBar";

describe("Visual Research Dashboard Chart Components", () => {
  afterEach(() => {
    cleanup();
  });

  describe("PassFailDonutChart", () => {
    it("renders empty state cleanly when total cases is 0", () => {
      render(<PassFailDonutChart passed={0} failed={0} executionErrors={0} />);
      expect(screen.getByText("N/A")).toBeDefined();
      expect(screen.getByText("Pass Rate")).toBeDefined();
      expect(screen.getByText("Total Test Cases")).toBeDefined();
    });

    it("renders 100% pass rate with correct percentage and aria-label", () => {
      const { container } = render(<PassFailDonutChart passed={10} failed={0} executionErrors={0} />);
      expect(screen.getByText("100.0%")).toBeDefined();
      expect(screen.getByText("Evaluation Passed")).toBeDefined();
      expect(screen.getByText("Evaluation Failed")).toBeDefined();

      const imgRole = container.querySelector('[role="img"]');
      expect(imgRole).not.toBeNull();
      expect(imgRole?.getAttribute("aria-label")).toContain("100.0%");
      expect(imgRole?.getAttribute("aria-label")).toContain("10 passed, 0 failed");
    });

    it("renders 0% pass rate when all tests fail", () => {
      const { container } = render(<PassFailDonutChart passed={0} failed={7} executionErrors={0} />);
      expect(screen.getByText("0.0%")).toBeDefined();
      expect(screen.getByText("Evaluation Passed")).toBeDefined();
      expect(screen.getByText("Evaluation Failed")).toBeDefined();

      const imgRole = container.querySelector('[role="img"]');
      expect(imgRole?.getAttribute("aria-label")).toContain("0.0%");
      expect(imgRole?.getAttribute("aria-label")).toContain("0 passed, 7 failed");
    });

    it("renders mixed distribution correctly with errors", () => {
      const { container } = render(<PassFailDonutChart passed={4} failed={4} executionErrors={2} />);
      expect(screen.getByText("40.0%")).toBeDefined();
      expect(screen.getByText("Evaluation Passed")).toBeDefined();
      expect(screen.getByText("Evaluation Failed")).toBeDefined();
      expect(screen.getByText("Execution Errors")).toBeDefined();

      const imgRole = container.querySelector('[role="img"]');
      expect(imgRole?.getAttribute("aria-label")).toContain("40.0%");
      expect(imgRole?.getAttribute("aria-label")).toContain("4 passed, 4 failed, 2 execution errors");
    });
  });

  describe("FailureDistributionChart", () => {
    it("renders clean 100% pass message when there are 0 failures", () => {
      render(<FailureDistributionChart categoryCounts={{}} totalFailures={0} />);
      expect(
        screen.getByText("No failures detected across evaluated test cases (100% pass rate)."),
      ).toBeDefined();
    });

    it("renders failure categories with formatted labels and explanations using Record", () => {
      const counts = {
        incorrect: 5,
        retrieval_failure: 2,
      };
      render(<FailureDistributionChart categoryCounts={counts} totalFailures={7} />);

      // Formatted readable labels
      expect(screen.getByText("Incorrect Answer")).toBeDefined();
      expect(screen.getByText("Retrieval Failure")).toBeDefined();

      // Explanations
      expect(
        screen.getByText(CATEGORY_HELP.incorrect),
      ).toBeDefined();
      expect(
        screen.getByText(CATEGORY_HELP.retrieval_failure),
      ).toBeDefined();
    });

    it("supports Map input format seamlessly", () => {
      const map = new Map<string, number>();
      map.set("unsupported", 3);
      map.set("contradicted", 1);

      render(<FailureDistributionChart categoryCounts={map} totalFailures={4} />);

      expect(screen.getByText("Unsupported Answer")).toBeDefined();
      expect(screen.getByText("Contradicted Answer")).toBeDefined();
      expect(
        screen.getByText(CATEGORY_HELP.unsupported),
      ).toBeDefined();
      expect(
        screen.getByText(CATEGORY_HELP.contradicted),
      ).toBeDefined();
    });
  });

  describe("VisualMetricsBar", () => {
    it("renders empty bar when total is 0", () => {
      const { container } = render(<VisualMetricsBar passed={0} failed={0} executionErrors={0} />);
      const bgBar = container.querySelector(".bg-paper-dim");
      expect(bgBar).not.toBeNull();
    });

    it("renders segments proportional to pass, fail, and error counts", () => {
      const { container } = render(
        <VisualMetricsBar passed={5} failed={3} executionErrors={2} showLabels={true} />,
      );
      expect(screen.getByText(/5 pass \(50%\)/)).toBeDefined();
      expect(screen.getByText(/3 fail \(30%\)/)).toBeDefined();
      expect(screen.getByText(/2 error/)).toBeDefined();
      expect(screen.getByText(/Total:\s*10/)).toBeDefined();

      const emeraldBar = container.querySelector(".bg-emerald-400");
      expect(emeraldBar).not.toBeNull();
      expect((emeraldBar as HTMLElement).style.width).toBe("50%");

      const redBar = container.querySelector(".bg-red-400");
      expect(redBar).not.toBeNull();
      expect((redBar as HTMLElement).style.width).toBe("30%");

      const amberBar = container.querySelector(".bg-amber-400");
      expect(amberBar).not.toBeNull();
      expect((amberBar as HTMLElement).style.width).toBe("20%");
    });
  });
});
