import type { ExperimentRun } from "../types/domain";

/**
 * Collapsed-by-default record of what identifies this run's conditions:
 * the configuration fingerprint (what was asked for) and the software
 * environment (what carried it out). Same fingerprint does not imply same
 * outputs: external and sampled models can still vary.
 */
export function ReproducibilityRecord({ run }: { run: ExperimentRun }) {
  const env = run.environment;
  const rows: [string, string][] = [
    ["Configuration fingerprint", run.config_fingerprint],
    ["Dataset version", run.dataset_version?.version_id ?? "n/a"],
    ["Adapter", run.adapter_id ?? "not recorded"],
    ["Repetition", String(run.repetition_index ?? 0)],
  ];
  if (env) {
    rows.push(
      ["Platform version", env.platform_version ?? "n/a"],
      [
        "Code (git commit)",
        env.git_commit
          ? `${env.git_commit.slice(0, 12)}${env.git_dirty_src ? " + uncommitted changes" : ""}`
          : "not a git checkout",
      ],
      ["Python", `${env.python ?? "n/a"} (${env.implementation ?? "?"}, ${env.os ?? "?"} ${env.machine ?? ""})`],
    );
    for (const [name, version] of Object.entries(env.dependencies ?? {})) rows.push([name, version]);
  }
  return (
    <details className="mt-4 text-sm">
      <summary className="cursor-pointer text-ink-soft hover:text-ink">Reproducibility record</summary>
      <dl className="mt-2 grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1 text-xs">
        {rows.map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="text-ink-soft">{k}</dt>
            <dd className="break-all font-mono text-ink">{v}</dd>
          </div>
        ))}
      </dl>
      {!env && (
        <p className="mt-2 text-xs text-ink-soft">
          This run was recorded before software versions were captured.
        </p>
      )}
      <p className="mt-2 text-xs text-ink-soft">
        The same fingerprint means the same configuration, not identical outputs: sampled or hosted
        models can still answer differently.
      </p>
    </details>
  );
}
