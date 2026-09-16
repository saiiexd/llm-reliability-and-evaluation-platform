import { Link, useParams } from "react-router-dom";
import { FixtureNotice } from "../components/FixtureNotice";
import { Card, SectionHeading } from "../components/Card";
import { EmptyState } from "../components/EmptyState";
import { Tag } from "../components/Tag";
import { toneForRunStatus } from "../lib/tone";
import { formatTimestamp, shortFingerprint } from "../lib/derived";
import { getExperiment, listRunsForExperiment } from "../data/dataAccess";

export function ExperimentDetailPage() {
  const { experimentId } = useParams<{ experimentId: string }>();
  const experiment = experimentId ? getExperiment(experimentId) : undefined;

  if (!experiment) {
    return <EmptyState title="Experiment not found." detail={`No experiment with id "${experimentId}".`} />;
  }

  const runs = listRunsForExperiment(experiment.experiment_id);
  const ragConfig = experiment.model_config.extra_params?.rag as Record<string, unknown> | undefined;

  return (
    <div>
      <SectionHeading eyebrow="Experiment definition" title={experiment.name} />
      <FixtureNotice />
      {experiment.description && <p className="mb-6 max-w-2xl text-sm">{experiment.description}</p>}

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <h3 className="mb-2 font-serif text-lg">Reproducibility</h3>
          <dl className="space-y-2 text-sm">
            <Row label="Experiment ID" value={<span className="font-mono">{experiment.experiment_id}</span>} />
            <Row
              label="Config fingerprint"
              value={<span className="font-mono">{shortFingerprint(experiment.config_fingerprint)}</span>}
            />
            <Row label="Created" value={formatTimestamp(experiment.created_at)} />
          </dl>
        </Card>

        <Card>
          <h3 className="mb-2 font-serif text-lg">Dataset</h3>
          <dl className="space-y-2 text-sm">
            <Row label="Name" value={experiment.dataset_version.dataset_name} />
            <Row
              label="Version ID"
              value={<span className="font-mono">{shortFingerprint(experiment.dataset_version.version_id)}</span>}
            />
            <Row label="Test cases" value={experiment.dataset_version.test_cases.length} />
          </dl>
        </Card>

        <Card>
          <h3 className="mb-2 font-serif text-lg">Model / application configuration</h3>
          <dl className="space-y-2 text-sm">
            <Row label="Model ID" value={<span className="font-mono">{experiment.model_config.model_id}</span>} />
            <Row label="Temperature" value={experiment.model_config.temperature ?? "not specified"} />
            <Row label="Max tokens" value={experiment.model_config.max_tokens ?? "not specified"} />
          </dl>
          {ragConfig && (
            <div className="mt-3 border-t border-line pt-3">
              <p className="mb-1 font-mono text-xs uppercase text-ink-soft">RAG configuration</p>
              <dl className="space-y-1 text-sm">
                <Row label="Embedding model" value={<span className="font-mono">{String(ragConfig.embedding_model_id)}</span>} />
                <Row label="Chunk size / overlap" value={`${ragConfig.chunk_size} / ${ragConfig.chunk_overlap}`} />
                <Row label="Top-K" value={String(ragConfig.top_k)} />
                <Row
                  label="Corpus fingerprint"
                  value={<span className="font-mono">{shortFingerprint(String(ragConfig.corpus_fingerprint))}</span>}
                />
              </dl>
            </div>
          )}
        </Card>

        <Card>
          <h3 className="mb-2 font-serif text-lg">Evaluator configuration</h3>
          <ul className="space-y-2 text-sm">
            {experiment.evaluator_configs.map((evaluatorConfig) => (
              <li key={evaluatorConfig.evaluator_name} className="border-b border-line pb-2 last:border-0">
                <span className="font-mono">{evaluatorConfig.evaluator_name}</span>
                {Object.keys(evaluatorConfig.parameters).length > 0 && (
                  <pre className="mt-1 overflow-x-auto bg-paper-dim p-2 font-mono text-xs">
                    {JSON.stringify(evaluatorConfig.parameters, null, 2)}
                  </pre>
                )}
              </li>
            ))}
          </ul>
        </Card>
      </div>

      <div className="mt-6">
        <Card>
          <SectionHeading eyebrow="Execution history" title="Runs" />
          {runs.length === 0 ? (
            <EmptyState title="This experiment has not been executed yet." />
          ) : (
            <ul className="divide-y divide-line">
              {runs.map((run) => (
                <li key={run.run_id} className="flex items-center justify-between py-2 text-sm">
                  <Link to={`/runs/${run.run_id}`} className="font-mono underline">
                    {run.run_id}
                  </Link>
                  <div className="flex items-center gap-3">
                    <span className="text-ink-soft">{formatTimestamp(run.started_at)}</span>
                    <Tag tone={toneForRunStatus(run.status)}>{run.status}</Tag>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4">
      <dt className="text-ink-soft">{label}</dt>
      <dd className="text-right">{value}</dd>
    </div>
  );
}
