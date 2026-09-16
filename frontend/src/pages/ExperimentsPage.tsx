import { Link } from "react-router-dom";
import { FixtureNotice } from "../components/FixtureNotice";
import { SectionHeading } from "../components/Card";
import { EmptyState } from "../components/EmptyState";
import { listExperiments, listRunsForExperiment } from "../data/dataAccess";
import { formatTimestamp } from "../lib/derived";

export function ExperimentsPage() {
  const experiments = listExperiments();

  return (
    <div>
      <SectionHeading eyebrow="Reproducibility" title="Experiments" />
      <FixtureNotice />

      {experiments.length === 0 ? (
        <EmptyState title="No experiments have been defined yet." />
      ) : (
        <div className="overflow-x-auto border border-line">
          <table className="w-full min-w-[720px] border-collapse text-sm">
            <thead>
              <tr className="border-b border-line bg-paper-dim text-left font-mono text-xs uppercase">
                <th className="px-3 py-2">Experiment</th>
                <th className="px-3 py-2">Dataset</th>
                <th className="px-3 py-2">Model</th>
                <th className="px-3 py-2">Evaluators</th>
                <th className="px-3 py-2">Created</th>
                <th className="px-3 py-2">Runs</th>
              </tr>
            </thead>
            <tbody>
              {experiments.map((experiment) => {
                const runs = listRunsForExperiment(experiment.experiment_id);
                return (
                  <tr key={experiment.experiment_id} className="border-b border-line last:border-0">
                    <td className="px-3 py-2">
                      <Link
                        to={`/experiments/${experiment.experiment_id}`}
                        className="font-medium underline"
                      >
                        {experiment.name}
                      </Link>
                      <div className="font-mono text-xs text-ink-soft">
                        {experiment.experiment_id}
                      </div>
                    </td>
                    <td className="px-3 py-2">
                      {experiment.dataset_version.dataset_name}
                      <div className="text-xs text-ink-soft">
                        {experiment.dataset_version.test_cases.length} test cases
                      </div>
                    </td>
                    <td className="px-3 py-2 font-mono text-xs">{experiment.model_config.model_id}</td>
                    <td className="px-3 py-2 font-mono text-xs">
                      {experiment.evaluator_configs.map((e) => e.evaluator_name).join(", ")}
                    </td>
                    <td className="px-3 py-2 text-xs text-ink-soft">
                      {formatTimestamp(experiment.created_at)}
                    </td>
                    <td className="px-3 py-2">{runs.length}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
