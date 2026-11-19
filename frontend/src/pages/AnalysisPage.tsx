import { useSearchParams } from "react-router-dom";
import { FixtureNotice } from "../components/FixtureNotice";
import { PageHeader } from "../components/Card";
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
    label: "Compare runs",
    tagline: "Did changing the setup change the results, or is the difference just noise?",
  },
  {
    id: "reliability",
    label: "Why answers fail",
    tagline: "Every non-passing answer, grouped by its diagnosed cause.",
  },
  {
    id: "retrieval",
    label: "Retrieval",
    tagline: "For RAG runs: did the search step find the passage that holds the answer?",
  },
  {
    id: "robustness",
    label: "Consistency",
    tagline: "Does the model give the same answer when the question is reworded or the passages reordered?",
  },
  {
    id: "performance",
    label: "Speed & cost",
    tagline: "How long answers take and how many tokens they use.",
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
    <div>
      <PageHeader
        title="Analysis"
        description="Look across runs instead of at one. Each tab answers one question."
      />

      <div role="tablist" className="mb-2 flex flex-wrap gap-1 border-b border-line pb-2">
        {ANALYSIS_TABS.map((tab) => {
          const isActive = tab.id === activeTab;
          return (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={isActive}
              onClick={() => setTab(tab.id)}
              className={`rounded-md px-3 py-1.5 text-sm transition-colors ${
                isActive ? "bg-paper-raised font-medium text-ink" : "text-ink-soft hover:text-ink"
              }`}
            >
              {tab.label}
            </button>
          );
        })}
      </div>
      <p className="mb-8 text-sm text-ink-soft">{currentTabMeta.tagline}</p>

      <FixtureNotice />

      {activeTab === "comparison" && <ExperimentComparisonPage />}
      {activeTab === "reliability" && <ReliabilityDiagnosticsPage />}
      {activeTab === "retrieval" && <RagDiagnosticsPage />}
      {activeTab === "robustness" && <RobustnessPage />}
      {activeTab === "performance" && <PerformanceAnalyticsPage />}
    </div>
  );
}
