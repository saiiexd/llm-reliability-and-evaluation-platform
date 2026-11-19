/**
 * "Do repeated runs agree?" (RQ4).
 *
 * Runs only count as repeats when the backend grouped them: same experiment,
 * same configuration fingerprint and same model adapter. Every rate is shown
 * as "x of y" so the denominator is always visible, and questions that could
 * not be compared (not answered in every run, or a check that was skipped or errored) are
 * reported as left out, never as agreeing.
 */
import { useConnection } from "../context/ConnectionContext";
import { useRepeatConsistency } from "../data/useData";
import { evaluatorLabel } from "../lib/derived";
import type { EvaluatorStability, RepeatGroupConsistency } from "../types/domain";
import { DataStateView } from "./DataStateView";
import { EmptyState } from "./EmptyState";
import { SimulatedNotice, SimulatedTag } from "./SimulatedNotice";
import { Tag } from "./Tag";

/** "57 of 60 (95%)", or "0 of 0" when there is nothing to compare. */
export function formatOfTotal(numerator: number, denominator: number): string {
  if (denominator <= 0) return `${numerator} of ${denominator}`;
  return `${numerator} of ${denominator} (${Math.round((numerator / denominator) * 100)}%)`;
}

function formatTemperature(t: number | null): string {
  return t === null ? "temperature not set" : `temperature ${t}`;
}

function IdList({ label, ids, total }: { label: string; ids: string[]; total: number }) {
  if (total === 0) return null;
  return (
    <details className="mt-2 text-xs">
      <summary className="cursor-pointer text-ink-soft">
        {label} ({total})
      </summary>
      <p className="mt-1 break-all font-mono text-ink-soft">
        {ids.join(", ")}
        {total > ids.length && ` … and ${total - ids.length} more`}
      </p>
    </details>
  );
}

function EvaluatorBar({ e }: { e: EvaluatorStability }) {
  const width = e.compared_cases > 0 ? (e.stable_cases / e.compared_cases) * 100 : 0;
  const reasons = [
    e.excluded_not_success > 0 && `${e.excluded_not_success} skipped or errored in a run`,
    e.excluded_no_verdict > 0 && `${e.excluded_no_verdict} gave no verdict in a run`,
    e.excluded_missing > 0 && `${e.excluded_missing} not checked in a run`,
  ].filter(Boolean);
  return (
    <li>
      <div className="flex items-baseline justify-between gap-4 text-sm">
        <span>{evaluatorLabel(e.evaluator_name)}</span>
        <span className="tabular-nums text-ink-soft">
          {formatOfTotal(e.stable_cases, e.compared_cases)}
        </span>
      </div>
      <div
        className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-paper-raised"
        role="img"
        aria-label={`${evaluatorLabel(e.evaluator_name)}: same verdict on ${formatOfTotal(e.stable_cases, e.compared_cases)}`}
      >
        <div className="h-full bg-ink-soft" style={{ width: `${width}%` }} />
      </div>
      {reasons.length > 0 && (
        <p className="mt-1 text-xs text-ink-soft">
          {e.excluded_cases} not comparable, left out: {reasons.join(", ")}.
        </p>
      )}
    </li>
  );
}

function Group({ g }: { g: RepeatGroupConsistency }) {
  return (
    <li className="py-5 first:pt-0 last:pb-0">
      <div className="flex flex-wrap items-center gap-2">
        <h4 className="font-medium">{g.experiment_name ?? g.experiment_id}</h4>
        <SimulatedTag run={g} />
        {!g.adapter_recorded && <Tag>Adapter not recorded</Tag>}
      </div>
      <p className="mt-0.5 text-xs text-ink-soft">
        {g.num_runs} runs · {formatTemperature(g.temperature)}
        {g.adapter_id && ` · ${g.adapter_id}`}
      </p>
      {!g.adapter_recorded && (
        <p className="mt-1 text-xs text-ink-soft">
          These older runs did not record which adapter answered, so they may not be true repeats.
        </p>
      )}

      <dl className="mt-4 grid gap-3 sm:grid-cols-2">
        <div>
          <dt className="text-xs text-ink-soft">Identical answers</dt>
          <dd className="text-lg font-semibold tabular-nums">
            {formatOfTotal(g.identical_output_cases, g.compared_cases)}
          </dd>
        </div>
        <div>
          <dt className="text-xs text-ink-soft">Same overall pass / fail</dt>
          <dd className="text-lg font-semibold tabular-nums">
            {formatOfTotal(g.case_pass_stable_cases, g.case_pass_compared_cases)}
          </dd>
        </div>
      </dl>
      <p className="mt-1 text-xs text-ink-soft">
        Identical means exactly the same text; a changed full stop counts as different.
        {g.excluded_cases > 0 &&
          ` ${g.excluded_cases} of ${g.total_test_cases} questions left out: not answered in every run.`}
      </p>

      {g.evaluators.length > 0 && (
        <>
          <p className="mt-4 text-xs text-ink-soft">Same verdict from each check in every run</p>
          <ul className="mt-2 space-y-3">
            {g.evaluators.map((e) => (
              <EvaluatorBar key={e.evaluator_name} e={e} />
            ))}
          </ul>
        </>
      )}

      <IdList label="Questions whose answer changed" ids={g.output_changed_test_case_ids} total={g.output_changed_count} />
      <IdList
        label="Questions whose overall pass / fail changed"
        ids={g.case_pass_changed_test_case_ids}
        total={g.case_pass_changed_count}
      />
      <SimulatedNotice run={g} />
    </li>
  );
}

export function RepeatConsistency({ experimentId }: { experimentId?: string }) {
  const { dataMode } = useConnection();
  const { data, loading, error, refetch } = useRepeatConsistency(experimentId);
  const groups = data?.groups ?? [];

  return (
    <section className="rounded-xl border border-line p-6">
      <h3 className="font-semibold">Do repeated runs agree?</h3>
      <p className="mt-0.5 text-sm text-ink-soft">
        Runs with the same experiment, setup and model, compared question by question.
      </p>
      <div className="mt-4">
        {dataMode !== "live" ? (
          <EmptyState
            title="Repeat analysis needs the live API."
            detail="The demo data does not include it. Connect to the backend, open an experiment and use “Compare models / repeat” to run the same model more than once."
          />
        ) : (
          <DataStateView loading={loading} error={error} onRetry={refetch}>
            {groups.length === 0 ? (
              <EmptyState
                title="No repeated runs yet."
                detail="Open an experiment, choose “Compare models / repeat” and repeat a model at least twice. Runs count as repeats only when the setup and model are identical."
              />
            ) : (
              <ul className="divide-y divide-line">
                {groups.map((g) => (
                  <Group key={`${g.experiment_id}-${g.config_fingerprint}-${g.adapter_id ?? "none"}`} g={g} />
                ))}
              </ul>
            )}
          </DataStateView>
        )}
      </div>
    </section>
  );
}
