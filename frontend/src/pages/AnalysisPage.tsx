import { useSearchParams } from "react-router-dom";
import { FixtureNotice } from "../components/FixtureNotice";
import { SectionHeading } from "../components/Card";
import { ExperimentComparisonPage } from "./ExperimentComparisonPage";
import { ReliabilityDiagnosticsPage } from "./ReliabilityDiagnosticsPage";
import { RagDiagnosticsPage } from "./RagDiagnosticsPage";
import { RobustnessPage } from "./RobustnessPage";
import { PerformanceAnalyticsPage } from "./PerformanceAnalyticsPage";

export type AnalysisTab = "comparison" | "reliability" | "retrieval" | "robustness" | "performance";

interface TabMeta {
  id: AnalysisTab;
  label: string;
  tagline: string;
}

const ANALYSIS_TABS: TabMeta[] = [
  {
    id: "comparison",
    label: "Model Comparison",
    tagline: "Side-by-side run comparisons with automated methodological compatibility verification.",
  },
  {
    id: "reliability",
    label: "Reliability & Failure Forensics",
    tagline: "Taxonomy of failure modes, unsupported claims, contradictions, and evaluator disagreements.",
  },
  {
    id: "retrieval",
    label: "RAG & Context Retrieval",
    tagline: "Retrieval accuracy metrics (Hit@1, MRR) and attribution of generation errors to context omissions.",
  },
  {
    id: "robustness",
    label: "Consistency & Robustness",
    tagline: "Stability of model outputs under input perturbations (question rewording, context reordering).",
  },
  {
    id: "performance",
    label: "System Latency & Telemetry",
    tagline: "Platform wall-clock timing, model latency distributions, and token usage accounting.",
  },
];

export function AnalysisPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const rawTab = searchParams.get("tab") as AnalysisTab | null;
  const activeTab: AnalysisTab = (rawTab && ANALYSIS_TABS.some((t) => t.id === rawTab))
    ? rawTab
    : "comparison";

  const setTab = (tab: AnalysisTab) => {
    setSearchParams({ tab });
  };

  const currentTabMeta = ANALYSIS_TABS.find((t) => t.id === activeTab) ?? ANALYSIS_TABS[0];

  return (
    <div className="space-y-6">
      {/* Analysis Section Header */}
      <div className="border-b border-line pb-4">
        <SectionHeading eyebrow="Research Investigation" title="Analytical Diagnostics" />
        <p className="mt-1 text-xs text-ink-soft max-w-2xl">
          Scientific investigation of model performance across controlled experimental conditions, failure taxonomy,
          retrieval effectiveness, and robustness perturbations.
        </p>

        {/* 5 Plain-Language Sub-Tabs */}
        <div className="mt-5 flex flex-wrap gap-2 border-b border-line pb-0">
          {ANALYSIS_TABS.map((tab) => {
            const isActive = tab.id === activeTab;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => setTab(tab.id)}
                className={`px-3 py-2 text-xs font-mono transition-colors border-b-2 -mb-[2px] ${
                  isActive
                    ? "border-ink text-ink font-semibold bg-paper-dim"
                    : "border-transparent text-ink-soft hover:text-ink hover:border-line"
                }`}
              >
                {tab.label}
              </button>
            );
          })}
        </div>
        <p className="mt-2 text-[11px] text-ink-soft italic">
          {currentTabMeta.tagline}
        </p>
      </div>

      <FixtureNotice />

      {/* Tab Content Rendering with 100% Functionality Preserved */}
      <div className="pt-2">
        {activeTab === "comparison" && <ExperimentComparisonPage />}
        {activeTab === "reliability" && <ReliabilityDiagnosticsPage />}
        {activeTab === "retrieval" && <RagDiagnosticsPage />}
        {activeTab === "robustness" && <RobustnessPage />}
        {activeTab === "performance" && <PerformanceAnalyticsPage />}
      </div>
    </div>
  );
}

