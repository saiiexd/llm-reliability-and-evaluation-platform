import { Link, useParams } from "react-router-dom";
import { FixtureNotice } from "../components/FixtureNotice";
import { Card, SectionHeading } from "../components/Card";
import { EmptyState } from "../components/EmptyState";
import { Tag } from "../components/Tag";
import {
  toneForClaimSupport,
  toneForDiagnosticStatus,
  toneForEvaluationStatus,
  toneForFailureCategory,
} from "../lib/tone";
import { formatScore, getRagMetadata, testCaseSucceeded } from "../lib/derived";
import { useDiagnostic, useRetrievalResults, useRun, useTestCase } from "../data/useData";
import { DataStateView } from "../components/DataStateView";

export function TestCaseDetailPage() {
  const { runId = "", testCaseId = "" } = useParams<{ runId: string; testCaseId: string }>();

  const { data: run, loading: runLoading, error: runError, refetch: refetchRun } = useRun(runId);
  const {
    data: tcResult,
    loading: tcLoading,
    error: tcError,
    refetch: refetchTc,
  } = useTestCase(runId, testCaseId);
  const {
    data: diag,
    loading: diagLoading,
    error: diagError,
    refetch: refetchDiag,
  } = useDiagnostic(runId, testCaseId);
  const { data: retrievalMap } = useRetrievalResults(runId);

  const loading = runLoading || tcLoading || diagLoading;
  const error = runError || tcError || diagError;
  const retrievalList = (retrievalMap && retrievalMap[testCaseId]) || [];

  const handleRetry = () => {
    void refetchRun();
    void refetchTc();
    void refetchDiag();
  };

  return (
    <div>
      <DataStateView loading={loading} error={error} onRetry={handleRetry}>
        {!run || !tcResult ? (
          <div>
            <SectionHeading eyebrow="Test case diagnosis" title="Test case not found" />
            <EmptyState
              title="Test case not found"
              detail={`No test case "${testCaseId}" found in run "${runId}".`}
            />
          </div>
        ) : (
          <>
            <div className="mb-2">
              <Link to={`/runs/${run.run_id}`} className="font-mono text-xs text-ink-soft underline">
                &larr; Back to Run {run.run_id.slice(0, 12)}
              </Link>
            </div>

            <SectionHeading
              eyebrow="Test case diagnosis"
              title={testCaseId}
              action={
                <div className="flex items-center gap-2">
                  <Tag tone={testCaseSucceeded(tcResult) ? "neutral" : "negative"}>
                    {testCaseSucceeded(tcResult) ? "execution succeeded" : "execution error"}
                  </Tag>
                </div>
              }
            />
            <FixtureNotice />

            {/* Primary & Secondary Diagnostic Summary */}
            <Card className="mb-6 border-l-4 border-l-ink">
              <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-xs uppercase text-ink-soft">Primary Category:</span>
                  {diag?.failure_category ? (
                    <Tag tone={toneForFailureCategory(diag.failure_category)}>
                      {diag.failure_category}
                    </Tag>
                  ) : (
                    <Tag tone="muted">undetermined</Tag>
                  )}
                </div>
                <div className="flex items-center gap-2 text-xs">
                  <span className="text-ink-soft">Diagnostic Status:</span>
                  <Tag tone={toneForDiagnosticStatus(diag?.status ?? "undetermined")}>
                    {diag?.status ?? "undetermined"}
                  </Tag>
                </div>
              </div>

              {diag?.explanation && (
                <p className="mt-3 text-sm font-medium leading-relaxed">{diag.explanation}</p>
              )}

              {/* Secondary Findings Section */}
              {diag?.secondary_findings && diag.secondary_findings.length > 0 && (
                <div className="mt-4 border-t border-line pt-3">
                  <p className="mb-2 font-mono text-xs uppercase tracking-wide text-ink-soft">
                    Preserved Secondary Observations ({diag.secondary_findings.length})
                  </p>
                  <ul className="space-y-1 text-xs">
                    {diag.secondary_findings.map((finding, idx) => (
                      <li key={idx} className="flex items-start gap-2 text-ink">
                        <span className="font-mono font-medium text-ink-soft">&bull;</span>
                        <span>{finding}</span>
                      </li>
                    ))}
                  </ul>
                  <p className="mt-2 text-[11px] text-ink-soft italic">
                    These findings were preserved alongside the primary diagnosis according to scientific
                    precedence rules, ensuring no failure evidence is masked.
                  </p>
                </div>
              )}
            </Card>

            {/* Prompt / Response Comparison */}
            <div className="mb-6 grid gap-6 lg:grid-cols-3">
              <Card>
                <SectionHeading eyebrow="Input" title="Question / Prompt" />
                <p className="whitespace-pre-wrap font-mono text-xs leading-relaxed">
                  {tcResult.input_text}
                </p>
              </Card>

              <Card>
                <SectionHeading eyebrow="Generated" title="Model Output" />
                {tcResult.execution_error ? (
                  <div className="bg-paper-dim p-2 font-mono text-xs text-accent">
                    {tcResult.execution_error}
                  </div>
                ) : (
                  <>
                    <p className="whitespace-pre-wrap font-mono text-xs leading-relaxed">
                      {tcResult.model_response?.output_text || tcResult.output_text || "(empty response)"}
                    </p>
                    {tcResult.model_response && (
                      <div className="mt-4 border-t border-line pt-3 text-[11px] font-mono text-ink-soft space-y-1">
                        <div className="flex justify-between">
                          <span>Model:</span>
                          <span className="text-ink">{tcResult.model_response.model_id}</span>
                        </div>
                        {tcResult.model_response.latency_ms !== null && (
                          <div className="flex justify-between">
                            <span>Latency (wall-clock):</span>
                            <span className="text-ink">{tcResult.model_response.latency_ms.toFixed(1)}ms</span>
                          </div>
                        )}
                        {(tcResult.model_response.prompt_tokens !== null || tcResult.model_response.completion_tokens !== null) && (
                          <div className="flex justify-between">
                            <span>Tokens (prompt / compl / total):</span>
                            <span className="text-ink font-semibold">
                              {tcResult.model_response.prompt_tokens ?? "—"} / {tcResult.model_response.completion_tokens ?? "—"} / {tcResult.model_response.total_tokens ?? "—"}
                            </span>
                          </div>
                        )}
                        {Boolean(tcResult.model_response.provider_metadata?.finish_reason) && (
                          <div className="flex justify-between">
                            <span>Finish Reason:</span>
                            <span className="text-ink">{String(tcResult.model_response.provider_metadata.finish_reason)}</span>
                          </div>
                        )}
                        {Boolean(tcResult.model_response.provider_metadata?.attempts) && (
                          <div className="flex justify-between">
                            <span>Attempts:</span>
                            <span className="text-ink">{String(tcResult.model_response.provider_metadata.attempts)}</span>
                          </div>
                        )}
                        {Boolean(tcResult.model_response.provider_metadata?.token_source) && (
                          <div className="flex justify-between">
                            <span>Token Provenance:</span>
                            <span className="text-ink font-sans">
                              <Tag tone="muted">{String(tcResult.model_response.provider_metadata.token_source)}</Tag>
                            </span>
                          </div>
                        )}
                      </div>
                    )}
                  </>
                )}
              </Card>

              <Card>
                <SectionHeading eyebrow="Ground truth" title="Reference Answer" />
                {tcResult.reference_answer ? (
                  <p className="whitespace-pre-wrap font-mono text-xs leading-relaxed">
                    {tcResult.reference_answer}
                  </p>
                ) : (
                  <p className="font-mono text-xs italic text-ink-soft">
                    No reference answer provided for this test case.
                  </p>
                )}
              </Card>
            </div>

            {/* Evaluators Results */}
            <div className="mb-6">
              <Card>
                <SectionHeading eyebrow="Evaluation" title="Evaluator Results" />
                {tcResult.evaluation_results.length === 0 ? (
                  <EmptyState title="No evaluators were run for this test case." />
                ) : (
                  <div className="overflow-x-auto border border-line">
                    <table className="w-full min-w-[640px] border-collapse text-sm">
                      <thead>
                        <tr className="border-b border-line bg-paper-dim text-left font-mono text-xs uppercase">
                          <th className="px-3 py-2">Evaluator</th>
                          <th className="px-3 py-2">Criterion</th>
                          <th className="px-3 py-2">Verdict</th>
                          <th className="px-3 py-2">Score</th>
                          <th className="px-3 py-2">Status</th>
                          <th className="px-3 py-2">Explanation</th>
                        </tr>
                      </thead>
                      <tbody>
                        {tcResult.evaluation_results.map((e, idx) => (
                          <tr key={idx} className="border-b border-line last:border-0">
                            <td className="px-3 py-2 font-mono text-xs">{e.evaluator_name}</td>
                            <td className="px-3 py-2 font-mono text-xs text-ink-soft">
                              {e.criterion ?? "n/a"}
                            </td>
                            <td className="px-3 py-2">
                              {e.passed === null ? (
                                <Tag tone="muted">n/a</Tag>
                              ) : (
                                <Tag tone={e.passed ? "neutral" : "negative"}>
                                  {e.passed ? "pass" : "fail"}
                                </Tag>
                              )}
                            </td>
                            <td className="px-3 py-2 font-mono text-xs">{formatScore(e.score)}</td>
                            <td className="px-3 py-2">
                              <Tag tone={toneForEvaluationStatus(e.status)}>{e.status}</Tag>
                            </td>
                            <td className="max-w-xs truncate px-3 py-2 text-xs text-ink-soft">
                              {e.explanation || e.error || "—"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </Card>
            </div>

            {/* Claim-Level Evidence Breakdown */}
            {diag?.claim_assessments && diag.claim_assessments.length > 0 && (
              <div className="mb-6">
                <Card>
                  <SectionHeading eyebrow="Forensic analysis" title="Claim-Level Faithfulness Breakdown" />
                  <p className="mb-3 text-xs text-ink-soft">
                    Sentences extracted from model output assessed for evidence support against
                    retrieved context.
                  </p>
                  <div className="overflow-x-auto border border-line">
                    <table className="w-full min-w-[680px] border-collapse text-sm">
                      <thead>
                        <tr className="border-b border-line bg-paper-dim text-left font-mono text-xs uppercase">
                          <th className="px-3 py-2">Claim / Statement</th>
                          <th className="px-3 py-2">Support Status</th>
                          <th className="px-3 py-2">Method</th>
                          <th className="px-3 py-2">Explanation / Evidence</th>
                        </tr>
                      </thead>
                      <tbody>
                        {diag.claim_assessments.map((ca, idx) => {
                          const claimStatus = ca.status ?? ca.assessment?.status ?? "unsupported";
                          const claimMethod = ca.method ?? ca.assessment?.method ?? "lexical";
                          const claimReasoning = ca.reasoning ?? ca.assessment?.explanation ?? "";
                          return (
                            <tr key={idx} className="border-b border-line last:border-0">
                              <td className="px-3 py-2 text-xs font-mono">{ca.statement}</td>
                              <td className="px-3 py-2">
                                <Tag tone={toneForClaimSupport(claimStatus)}>{claimStatus}</Tag>
                              </td>
                              <td className="px-3 py-2 font-mono text-xs text-ink-soft">{String(claimMethod)}</td>
                              <td className="px-3 py-2 text-xs text-ink-soft">{claimReasoning}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </Card>
              </div>
            )}

            {/* RAG Retrieved Context & Traces */}
            {getRagMetadata(tcResult.model_response) && (
              <div className="mb-6">
                <Card>
                  <SectionHeading eyebrow="Retrieval telemetry" title="Retrieved Passages & Context" />
                  {(() => {
                    const ragMeta = getRagMetadata(tcResult.model_response);
                    if (!ragMeta) return null;
                    return (
                      <>
                        <div className="mb-3 flex items-center justify-between font-mono text-xs">
                          <span>
                            Query: <code className="bg-paper-dim px-1">{ragMeta.query}</code>
                          </span>
                          <span>Retrieved chunks: {ragMeta.retrieved_chunks.length}</span>
                        </div>

                        <div className="space-y-3">
                          {ragMeta.retrieved_chunks.map((chunk, idx) => (
                            <div key={idx} className="border border-line bg-paper-dim p-3 text-xs">
                              <div className="mb-1 flex items-center justify-between font-mono text-ink-soft">
                                <span>
                                  Rank {chunk.rank} | Document: {chunk.document_id}
                                </span>
                                <span>Similarity score: {formatScore(chunk.score)}</span>
                              </div>
                              <p className="whitespace-pre-wrap font-mono text-ink">{chunk.text}</p>
                            </div>
                          ))}
                        </div>
                      </>
                    );
                  })()}
                </Card>
              </div>
            )}

            {/* Retrieval Metrics if present */}
            {retrievalList.length > 0 && (
              <div className="mb-6">
                <Card>
                  <SectionHeading eyebrow="Metrics" title="Retrieval Evaluation Metrics" />
                  <div className="overflow-x-auto border border-line">
                    <table className="w-full min-w-[500px] border-collapse text-sm">
                      <thead>
                        <tr className="border-b border-line bg-paper-dim text-left font-mono text-xs uppercase">
                          <th className="px-3 py-2">Metric</th>
                          <th className="px-3 py-2">Score</th>
                          <th className="px-3 py-2">K</th>
                          <th className="px-3 py-2">Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {retrievalList.map((rm, idx) => (
                          <tr key={idx} className="border-b border-line last:border-0">
                            <td className="px-3 py-2 font-mono text-xs">{rm.metric_name}</td>
                            <td className="px-3 py-2 font-mono text-xs">{formatScore(rm.score)}</td>
                            <td className="px-3 py-2 font-mono text-xs">
                              {String(rm.k ?? rm.evaluator_config?.k ?? "n/a")}
                            </td>
                            <td className="px-3 py-2">
                              <Tag tone={toneForEvaluationStatus(rm.status)}>{rm.status}</Tag>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>
              </div>
            )}
          </>
        )}
      </DataStateView>
    </div>
  );
}
