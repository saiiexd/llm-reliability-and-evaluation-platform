"""
Analyze EXP-002 runs and write results.json, results_tables.md and per_case.csv.

Implements the analysis plan fixed in
research/experiments/EXP-002_robustness/definition.md before the run:

- reference = ``base`` run 1; noise floor = ``base`` run 2 vs run 1;
- per arm and evaluator: pass rate (Wilson 95%), paired verdict flips vs the
  reference split by direction, flip rate (Wilson 95%), exact McNemar p;
- identical-output rate vs the reference (SQuAD-normalized text);
- rule-based diagnosis consistency via ``diagnostics.robustness``;
- retrieval: whether the retrieved chunk ids equal the reference's, and
  Hit@3 / Recall@3 / Precision@3 / MRR;
- paraphrase arm: primary analysis excludes paraphrases reviewed as
  ``meaning_changed``; sensitivity analysis uses all questions;
- latency and token usage.

Nothing is re-run; all numbers come from persisted runs.

Usage:
    python scripts/analyze_robustness_study.py [--store experiments]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_reliability_study import (  # noqa: E402
    fmt_p,
    mcnemar_exact,
    pct,
    verdict,
    wilson,
)

from llm_reliability.api.services import DiagnosticsService  # noqa: E402
from llm_reliability.diagnostics.robustness import (  # noqa: E402
    PerturbationType,
    compare_robustness,
    summarize_robustness,
)
from llm_reliability.evaluation.squad_match import squad_normalize  # noqa: E402
from llm_reliability.experiments import LocalFileExperimentRepository  # noqa: E402
from llm_reliability.experiments.models import ExperimentRun  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "research" / "experiments" / "EXP-002_robustness"
EVALUATORS = ["exact_match", "squad_match", "llm_judge", "llm_faithfulness_judge"]
PRIMARY = "llm_judge"
ARMS = ["reversed", "distractor", "paraphrase", "freeform"]
PERTURBATION = {
    "noise": PerturbationType.REPEATED_INPUT,
    "reversed": PerturbationType.CONTEXT_REORDERING,
    "distractor": PerturbationType.IRRELEVANT_CONTEXT_INJECTION,
    "paraphrase": PerturbationType.QUESTION_REWORDING,
    "freeform": PerturbationType.PROMPT_TEMPLATE_CHANGE,
}
ABSTAIN = "i dont know"


def by_case(run: ExperimentRun) -> dict[str, Any]:
    assert run.result is not None
    return {r.test_case_id: r for r in run.result.test_case_results}


def compare(
    ref: ExperimentRun,
    other: ExperimentRun,
    key: str,
    diagnostics: DiagnosticsService,
    keep: set[str] | None = None,
) -> dict[str, Any]:
    a, b = by_case(ref), by_case(other)
    ids = [i for i in a if i in b and (keep is None or i in keep)]
    out: dict[str, Any] = {"n_questions": len(ids), "evaluators": {}}
    for ev in EVALUATORS:
        pairs = [(verdict(a[i], ev), verdict(b[i], ev)) for i in ids]
        scored = [(x, y) for x, y in pairs if x is not None and y is not None]
        a_only = sum(1 for x, y in scored if x and not y)
        b_only = sum(1 for x, y in scored if not x and y)
        flips = a_only + b_only
        out["evaluators"][ev] = {
            "n_paired": len(scored),
            "excluded_not_scored": len(pairs) - len(scored),
            "ref_pass": sum(1 for x, _ in scored if x),
            "arm_pass": sum(1 for _, y in scored if y),
            "ref_pass_rate": wilson(sum(1 for x, _ in scored if x), len(scored)),
            "arm_pass_rate": wilson(sum(1 for _, y in scored if y), len(scored)),
            "flips_pass_to_fail": a_only,
            "flips_fail_to_pass": b_only,
            "flip_rate": wilson(flips, len(scored)),
            "mcnemar_exact_p": mcnemar_exact(a_only, b_only),
        }
    outs = [
        (
            squad_normalize(a[i].model_response.output_text) if a[i].model_response else None,
            squad_normalize(b[i].model_response.output_text) if b[i].model_response else None,
        )
        for i in ids
    ]
    out["identical_outputs"] = sum(1 for x, y in outs if x is not None and x == y)
    out["abstentions_ref"] = sum(1 for x, _ in outs if x == ABSTAIN)
    out["abstentions_arm"] = sum(1 for _, y in outs if y == ABSTAIN)

    def retrieved(tc: Any) -> list[str] | None:
        rag = tc.model_response.provider_metadata.get("rag") if tc.model_response else None
        return [c["chunk_id"] for c in rag["retrieved_chunks"]] if rag else None

    out["identical_retrieval"] = sum(1 for i in ids if retrieved(a[i]) == retrieved(b[i]))

    da, db = diagnostics.get_diagnostics(ref), diagnostics.get_diagnostics(other)
    results = [
        compare_robustness(da[i], db[i], PERTURBATION[key]) for i in ids if i in da and i in db
    ]
    summary = summarize_robustness(results)
    out["diagnosis_consistency"] = summary.to_dict()
    out["changed_judge_ids"] = [
        i
        for i in ids
        if verdict(a[i], PRIMARY) is not None
        and verdict(b[i], PRIMARY) is not None
        and verdict(a[i], PRIMARY) != verdict(b[i], PRIMARY)
    ]
    return out


def run_stats(run: ExperimentRun, diagnostics: DiagnosticsService) -> dict[str, Any]:
    assert run.result is not None
    tcs = run.result.test_case_results
    lat = [
        r.model_response.latency_ms for r in tcs if r.model_response and r.model_response.latency_ms
    ]
    ptok = [
        r.model_response.prompt_tokens
        for r in tcs
        if r.model_response and r.model_response.prompt_tokens
    ]
    ctok = [
        r.model_response.completion_tokens
        for r in tcs
        if r.model_response and r.model_response.completion_tokens
    ]
    metrics: dict[str, list[float]] = {}
    for results in diagnostics.get_retrieval_results(run).values():
        for res in results:
            if res.score is not None:
                metrics.setdefault(res.metric_name, []).append(res.score)
    return {
        "run_id": run.run_id,
        "repetition_index": run.repetition_index,
        "n": len(tcs),
        "pass_rates": {
            ev: wilson(
                sum(1 for r in tcs if verdict(r, ev)),
                sum(1 for r in tcs if verdict(r, ev) is not None),
            )
            for ev in EVALUATORS
        },
        "retrieval_means": {m: statistics.fmean(v) for m, v in sorted(metrics.items())},
        "latency_ms_median": statistics.median(lat) if lat else None,
        "prompt_tokens_mean": statistics.fmean(ptok) if ptok else None,
        "completion_tokens_mean": statistics.fmean(ctok) if ctok else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=REPO_ROOT / "experiments")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--pilot", action="store_true", help="analyze -pilotN experiments only")
    args = parser.parse_args()
    repo = LocalFileExperimentRepository(args.store)
    runs: dict[str, list[ExperimentRun]] = {}
    for exp_id in repo.list_experiment_ids():
        if not exp_id.startswith("exp002-") or ("-pilot" in exp_id) != args.pilot:
            continue
        arm = exp_id.removeprefix("exp002-").split("-pilot")[0]
        runs[arm] = sorted(
            (repo.load_run(r) for r in repo.list_run_ids(exp_id)), key=lambda r: r.repetition_index
        )
    if len(runs.get("base", [])) < 2:
        raise SystemExit("EXP-002 needs two base runs (noise floor); found fewer.")
    ref, noise = runs["base"][0], runs["base"][1]
    diagnostics = DiagnosticsService()

    para_raw = json.loads(
        (REPO_ROOT / "data/benchmarks/squad_v1_1_dev_subset_v1_paraphrases_v1.json").read_text()
    )
    same_meaning = {
        it["id"] for it in para_raw["items"] if it["meaning_review"]["status"] != "meaning_changed"
    }

    report: dict[str, Any] = {
        "study": "EXP-002",
        "reference_run": ref.run_id,
        "environment": repo.load_experiment(ref.experiment_id).metadata.get("study"),
        "runs": {arm: [run_stats(r, diagnostics) for r in rs] for arm, rs in runs.items()},
        "contrasts": {"noise_floor (base run 2)": compare(ref, noise, "noise", diagnostics)},
    }
    for arm in ARMS:
        if arm not in runs:
            continue
        if arm == "paraphrase":
            report["contrasts"]["paraphrase (primary: meaning preserved)"] = compare(
                ref, runs[arm][0], arm, diagnostics, keep=same_meaning
            )
            report["contrasts"]["paraphrase (sensitivity, all 60)"] = compare(
                ref, runs[arm][0], arm, diagnostics
            )
        else:
            report["contrasts"][arm] = compare(ref, runs[arm][0], arm, diagnostics)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "results_tables.md").write_text(render(report))
    write_per_case(runs, args.out / "per_case.csv")
    print(render(report))
    return 0


def write_per_case(runs: dict[str, list[ExperimentRun]], path: Path) -> None:
    columns = [("base1", runs["base"][0]), ("base2", runs["base"][1])] + [
        (arm, runs[arm][0]) for arm in ARMS if arm in runs
    ]
    cases = {k: by_case(r) for k, r in columns}
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        header = ["test_case_id", "question"]
        for k, _ in columns:
            header += [f"{k}_output", f"{k}_judge", f"{k}_squad_em"]
        w.writerow(header)
        for tc_id, tc in cases["base1"].items():
            row = [tc_id, tc.input_text]
            for k, _ in columns:
                c = cases[k].get(tc_id)
                row += [
                    c.model_response.output_text if c and c.model_response else "",
                    verdict(c, "llm_judge") if c else "",
                    verdict(c, "squad_match") if c else "",
                ]
            w.writerow(row)


def render(report: dict[str, Any]) -> str:
    lines = [
        "# EXP-002 result tables (generated by scripts/analyze_robustness_study.py)",
        "",
        f"Reference: base run 1 (`{report['reference_run']}`). Each contrast pairs the same "
        "questions. Flip = the verdict differs from the reference.",
        "",
        "## Primary measure: LLM-judge verdict flips vs the reference",
        "",
        "| Contrast | n | Judge pass ref → arm | Flips (pass→fail / fail→pass) | Flip rate [95% CI] | McNemar p | Identical outputs | Identical retrieval | Diagnosis unchanged |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, c in report["contrasts"].items():
        e = c["evaluators"][PRIMARY]
        fr = e["flip_rate"]
        d = c["diagnosis_consistency"]
        lines.append(
            f"| {name} | {e['n_paired']} | {e['ref_pass']} → {e['arm_pass']} | "
            f"{e['flips_pass_to_fail']} / {e['flips_fail_to_pass']} | "
            f"{pct(fr[0])} [{pct(fr[1])}, {pct(fr[2])}] | {fmt_p(e['mcnemar_exact_p'])} | "
            f"{c['identical_outputs']}/{c['n_questions']} | "
            f"{c['identical_retrieval']}/{c['n_questions']} | "
            f"{d['num_unchanged']}/{d['num_comparisons']} |"
        )
    lines += [
        "",
        "## All evaluators: pass counts and flips vs the reference",
        "",
        "| Contrast | Evaluator | n | Pass ref → arm | Flips (→fail / →pass) | McNemar p |",
        "|---|---|---|---|---|---|",
    ]
    for name, c in report["contrasts"].items():
        for ev, e in c["evaluators"].items():
            lines.append(
                f"| {name} | {ev} | {e['n_paired']} | {e['ref_pass']} → {e['arm_pass']} | "
                f"{e['flips_pass_to_fail']} / {e['flips_fail_to_pass']} | "
                f"{fmt_p(e['mcnemar_exact_p'])} |"
            )
    lines += [
        "",
        "## Per run: pass rates, retrieval, cost",
        "",
        "| Arm | Rep | Exact | SQuAD EM | Judge correct [95% CI] | Faithful | Hit@3 | Precision@3 | MRR | Median latency (ms) | Prompt tokens | Completion tokens |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, stats in report["runs"].items():
        for s in stats:
            pr, rm = s["pass_rates"], s["retrieval_means"]
            j = pr["llm_judge"]
            lines.append(
                f"| {arm} | {s['repetition_index']} | {pct(pr['exact_match'][0])} | "
                f"{pct(pr['squad_match'][0])} | {pct(j[0])} [{pct(j[1])}, {pct(j[2])}] | "
                f"{pct(pr['llm_faithfulness_judge'][0])} | {pct(rm.get('hit_at_k'))} | "
                f"{pct(rm.get('precision_at_k'))} | {rm.get('mrr', 0):.3f} | "
                f"{s['latency_ms_median']:.0f} | {s['prompt_tokens_mean']:.0f} | "
                f"{s['completion_tokens_mean']:.1f} |"
            )
    lines += ["", "## Questions whose judge verdict changed", ""]
    for name, c in report["contrasts"].items():
        lines.append(f"- {name}: {', '.join(c['changed_judge_ids']) or 'none'}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
