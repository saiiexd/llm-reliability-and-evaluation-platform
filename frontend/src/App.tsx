import { Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { OverviewPage } from "./pages/OverviewPage";
import { ExperimentsPage } from "./pages/ExperimentsPage";
import { ExperimentDetailPage } from "./pages/ExperimentDetailPage";
import { ResultsPage } from "./pages/ResultsPage";
import { AnalysisPage } from "./pages/AnalysisPage";
import { NotFoundPage } from "./pages/NotFoundPage";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        {/* 1. Overview */}
        <Route index element={<OverviewPage />} />

        {/* 2. Experiments */}
        <Route path="experiments" element={<ExperimentsPage />} />
        <Route path="experiments/:experimentId" element={<ExperimentDetailPage />} />

        {/* 3. Results */}
        <Route path="results" element={<ResultsPage />} />
        <Route path="results/:runId" element={<ResultsPage />} />
        {/* Legacy run links seamlessly redirect/alias to Results */}
        <Route path="runs/:runId" element={<ResultsPage />} />
        <Route path="runs/:runId/test-cases/:testCaseId" element={<ResultsPage />} />

        {/* 4. Analysis */}
        <Route path="analysis" element={<AnalysisPage />} />
        {/* Legacy analysis links alias directly to their contextual tab */}
        <Route path="comparisons" element={<Navigate to="/analysis?tab=comparison" replace />} />
        <Route path="analytics" element={<Navigate to="/analysis?tab=comparison" replace />} />
        <Route path="reliability" element={<Navigate to="/analysis?tab=reliability" replace />} />
        <Route path="failures" element={<Navigate to="/analysis?tab=reliability" replace />} />
        <Route path="rag" element={<Navigate to="/analysis?tab=retrieval" replace />} />
        <Route path="evaluators" element={<Navigate to="/analysis?tab=reliability" replace />} />
        <Route path="robustness" element={<Navigate to="/analysis?tab=robustness" replace />} />
        <Route path="performance" element={<Navigate to="/analysis?tab=performance" replace />} />

        {/* Catch-all */}
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
}
