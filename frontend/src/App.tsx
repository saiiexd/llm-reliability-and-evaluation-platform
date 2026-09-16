import { Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { OverviewPage } from "./pages/OverviewPage";
import { ExperimentsPage } from "./pages/ExperimentsPage";
import { ExperimentDetailPage } from "./pages/ExperimentDetailPage";
import { RunDetailPage } from "./pages/RunDetailPage";
import { TestCaseDetailPage } from "./pages/TestCaseDetailPage";
import { RagDiagnosticsPage } from "./pages/RagDiagnosticsPage";
import { EvaluatorComparisonPage } from "./pages/EvaluatorComparisonPage";
import { ReliabilityDiagnosticsPage } from "./pages/ReliabilityDiagnosticsPage";
import { RobustnessPage } from "./pages/RobustnessPage";
import { NotFoundPage } from "./pages/NotFoundPage";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<OverviewPage />} />
        <Route path="experiments" element={<ExperimentsPage />} />
        <Route path="experiments/:experimentId" element={<ExperimentDetailPage />} />
        <Route path="runs/:runId" element={<RunDetailPage />} />
        <Route path="runs/:runId/test-cases/:testCaseId" element={<TestCaseDetailPage />} />
        <Route path="rag" element={<RagDiagnosticsPage />} />
        <Route path="evaluators" element={<EvaluatorComparisonPage />} />
        <Route path="reliability" element={<ReliabilityDiagnosticsPage />} />
        <Route path="robustness" element={<RobustnessPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
}
