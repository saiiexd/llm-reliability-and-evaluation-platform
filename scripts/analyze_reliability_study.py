"""
Analyze EXP-001 runs and write results.json + results_tables.md.

Reads every persisted run of the ``exp001-*`` experiments from a store and
computes, without re-running any model:

- per arm and evaluator: pass rate with a Wilson 95% interval (n = cases the
  evaluator actually scored), SQuAD token-F1 mean, judge label counts,
  abstention ("I don't know") rate;
- retrieval at each run's own K: Hit@K / Recall@K / MRR against the strict
  SQuAD source-paragraph labels, plus a lenient "answer-containment" Hit@K
  (any retrieved chunk contains a gold answer string after normalization);
- correctness conditioned on retrieval hit vs miss (RQ2);
- paired comparisons on the same questions (exact McNemar test) for the
  pre-registered contrasts;
- evaluator agreement (raw and Cohen's kappa) between correctness evaluators;
- rule-based failure categories (llm_reliability.diagnostics);
- determinism (two T=0 runs) and consistency (three T=0.8 runs);
- generation latency and token usage.

Usage:
    python scripts/analyze_reliability_study.py [--store experiments]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from llm_reliability.analytics.agreement import compute_all_pairwise_agreements
from llm_reliability.api.services import _run_top_k
from llm_reliability.diagnostics.diagnosis import build_reliability_diagnostic
from llm_reliability.evaluation.models import EvaluationStatus, TestCaseResult
from llm_reliability.evaluation.squad_match import gold_answers_for, squad_normalize
from llm_reliability.experiments import LocalFileExperimentRepository
from llm_reliability.experiments.models import ExperimentRun
from llm_reliability.rag.evaluators import (
    HitAtKEvaluator,
    MeanReciprocalRankEvaluator,
    RecallAtKEvaluator,
    evaluate_retrieval,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "research" / "experiments" / "EXP-001_local_rag_reliability"
CORRECTNESS_EVALUATORS = ["exact_match", "squad_match", "llm_judge"]
CONTRASTS = [
    ("qwen-closed", "qwen-rag-k3", "Retrieval effect, qwen3:4b (closed book vs top-3)"),
    ("qwen-rag-k1", "qwen-rag-k3", "Retrieval depth, qwen3:4b (top-1 vs top-3)"),
    ("gemma-closed", "gemma-rag-k3", "Retrieval effect, gemma3:1b (closed book vs top-3)"),
    ("gemma-rag-k3", "qwen-rag-k3", "Model effect with top-3 retrieval (gemma3:1b vs qwen3:4b)"),
]
ABSTAIN = "i dont know"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None, float | None]:
    if n == 0:
        return None, None, None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (binomial test on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def verdict(tc: TestCaseResult, evaluator: str) -> bool | None:
    for e in tc.evaluation_results:
        if e.evaluator_name == evaluator:
            return e.passed if e.status == EvaluationStatus.SUCCESS else None
    return None


def arm_of(run: ExperimentRun) -> str:
    return run.experiment_id.removeprefix("exp001-").split("-pilot")[0]


def lenient_hit(tc_meta_gold: list[str], retrieved_texts: list[str]) -> bool:
    golds = [squad_normalize(g) for g in tc_meta_gold if squad_normalize(g)]
    texts = [squad_normalize(t) for t in retrieved_texts]
    return any(g in t for g in golds for t in texts)


def summarize_run(run: ExperimentRun) -> dict[str, Any]:
    assert run.result is not None
    cases = run.dataset_version.dataset.test_cases
    by_id = {tc.id: tc for tc in cases}
    results = run.result.test_case_results
    out: dict[str, Any] = {
        "run_id": run.run_id,
        "arm": arm_of(run),
        "repetition_index": run.repetition_index,
        "adapter_id": run.adapter_id,
        "executed_model_ids": run.executed_model_ids,
        "simulated_components": run.simulated_components,
        "n_cases": len(results),
        "n_execution_errors": sum(1 for r in results if not r.succeeded),
        "evaluators": {},
    }
    for name in run.result.evaluator_names:
        evs = [e for r in results for e in r.evaluation_results if e.evaluator_name == name]
        scored = [e for e in evs if e.status == EvaluationStatus.SUCCESS and e.passed is not None]
        k = sum(1 for e in scored if e.passed)
        p, lo, hi = wilson(k, len(scored))
        out["evaluators"][name] = {
            "passed": k,
            "scored": len(scored),
            "pass_rate": p,
            "ci95": [lo, hi],
            "status_counts": dict(Counter(e.status.value for e in evs)),
            "label_counts": dict(Counter(e.label for e in evs if e.label is not None)),
            "score_mean": statistics.fmean([e.score for e in evs if e.score is not None])
            if any(e.score is not None for e in evs)
            else None,
        }
    outputs = [r.model_response.output_text for r in results if r.model_response]
    out["abstention_rate"] = (
        sum(1 for o in outputs if squad_normalize(o) == ABSTAIN) / len(outputs) if outputs else None
    )
    latencies = [
        r.model_response.latency_ms
        for r in results
        if r.model_response and r.model_response.latency_ms
    ]
    tokens = [
        r.model_response.total_tokens
        for r in results
        if r.model_response and r.model_response.total_tokens
    ]
    out["latency_ms"] = (
        {
            "median": statistics.median(latencies),
            "p90": statistics.quantiles(latencies, n=10)[-1]
            if len(latencies) >= 10
            else max(latencies),
            "max": max(latencies),
        }
        if latencies
        else None
    )
    out["total_tokens_mean"] = statistics.fmean(tokens) if tokens else None

    # Retrieval at the run's own K (strict labels) + lenient answer containment.
    responses = {r.test_case_id: r.model_response for r in results if r.model_response}
    is_rag = any("rag" in m.provider_metadata for m in responses.values())
    retrieval_by_case: dict[str, list[Any]] = {}
    if is_rag:
        k_run = _run_top_k(responses.values())
        retrieval_by_case = evaluate_retrieval(
            cases,
            responses,
            [HitAtKEvaluator(k_run), RecallAtKEvaluator(k_run), MeanReciprocalRankEvaluator()],
        )
        metric_scores: dict[str, list[float]] = {}
        for res_list in retrieval_by_case.values():
            for res in res_list:
                if res.score is not None:
                    metric_scores.setdefault(res.metric_name, []).append(res.score)
        lenient = [
            lenient_hit(
                gold_answers_for(by_id[tc_id]),
                [c["text"] for c in m.provider_metadata["rag"]["retrieved_chunks"]],
            )
            for tc_id, m in responses.items()
        ]
        strict_hit = {
            tc_id: next((x.score > 0 for x in lst if x.metric_name == "hit_at_k"), None)
            for tc_id, lst in retrieval_by_case.items()
        }
        judge_given = {True: [], False: []}  # type: dict[bool, list[bool]]
        for r in results:
            h, v = strict_hit.get(r.test_case_id), verdict(r, "llm_judge")
            if h is not None and v is not None:
                judge_given[h].append(v)
        out["retrieval"] = {
            "k": k_run,
            "strict_mean": {m: statistics.fmean(v) for m, v in metric_scores.items()},
            "lenient_answer_containment_hit_rate": statistics.fmean(lenient) if lenient else None,
            "judge_correct_given_hit": wilson(sum(judge_given[True]), len(judge_given[True])),
            "judge_correct_given_miss": wilson(sum(judge_given[False]), len(judge_given[False])),
            "n_hit": len(judge_given[True]),
            "n_miss": len(judge_given[False]),
        }

    # Rule-based failure categories (same function the API uses).
    categories = Counter()
    for r in results:
        diag = build_reliability_diagnostic(
            by_id[r.test_case_id], r, retrieval_by_case.get(r.test_case_id, [])
        )
        categories[diag.failure_category.value if diag.failure_category else diag.status.value] += 1
    out["failure_categories"] = dict(categories)

    out["agreement"] = [
        {
            "pair": f"{a.evaluator_a} vs {a.evaluator_b}",
            "cases": a.cases_compared,
            "agreement_rate": a.agreement_rate,
            "cohens_kappa": a.cohens_kappa,
            "table_both_pass_a_only_b_only_both_fail": [
                a.both_passed,
                a.only_a_passed,
                a.only_b_passed,
                a.both_failed,
            ],
        }
        for a in compute_all_pairwise_agreements(run)
        if a.evaluator_a in CORRECTNESS_EVALUATORS and a.evaluator_b in CORRECTNESS_EVALUATORS
    ]
    return out


def paired(run_a: ExperimentRun, run_b: ExperimentRun, evaluator: str) -> dict[str, Any]:
    assert run_a.dataset_version.version_id == run_b.dataset_version.version_id
    va = {r.test_case_id: verdict(r, evaluator) for r in run_a.result.test_case_results}  # type: ignore[union-attr]
    vb = {r.test_case_id: verdict(r, evaluator) for r in run_b.result.test_case_results}  # type: ignore[union-attr]
    both = [(va[i], vb[i]) for i in va if va[i] is not None and vb.get(i) is not None]
    b = sum(1 for x, y in both if x and not y)  # a right, b wrong
    c = sum(1 for x, y in both if not x and y)  # a wrong, b right
    return {
        "evaluator": evaluator,
        "n_paired": len(both),
        "a_pass": sum(1 for x, _ in both if x),
        "b_pass": sum(1 for _, y in both if y),
        "a_only": b,
        "b_only": c,
        "mcnemar_exact_p": mcnemar_exact(b, c),
    }


def outputs_of(run: ExperimentRun) -> dict[str, str]:
    return {
        r.test_case_id: squad_normalize(r.model_response.output_text)
        for r in run.result.test_case_results
        if r.model_response
    }  # type: ignore[union-attr]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=REPO_ROOT / "experiments")
    parser.add_argument("--pilot", action="store_true", help="analyze *-pilotN experiments instead")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    repo = LocalFileExperimentRepository(args.store)
    exp_ids = sorted(
        e
        for e in repo.list_experiment_ids()
        if e.startswith("exp001-") and (("-pilot" in e) == args.pilot)
    )
    runs: dict[str, list[ExperimentRun]] = {}
    exp_for_arm: dict[str, str] = {}
    for exp_id in exp_ids:
        loaded = [repo.load_run(r) for r in repo.list_run_ids(exp_id)]
        arm = exp_id.removeprefix("exp001-").split("-pilot")[0]
        runs[arm] = sorted(loaded, key=lambda r: r.repetition_index)
        exp_for_arm[arm] = exp_id

    report: dict[str, Any] = {"arms": {}, "contrasts": [], "determinism": None, "consistency": None}
    for arm, arm_runs in runs.items():
        report["arms"][arm] = [summarize_run(r) for r in arm_runs]
        definition = repo.load_experiment(exp_for_arm[arm])
        report.setdefault("environment", definition.metadata.get("study"))

    for a, b, label in CONTRASTS:
        if a in runs and b in runs:
            report["contrasts"].append(
                {
                    "label": label,
                    "a": a,
                    "b": b,
                    "results": [
                        paired(runs[a][0], runs[b][0], ev) for ev in ("llm_judge", "squad_match")
                    ],
                }
            )

    if len(runs.get("qwen-rag-k3", [])) >= 2:
        o1, o2 = outputs_of(runs["qwen-rag-k3"][0]), outputs_of(runs["qwen-rag-k3"][1])
        same = sum(1 for i in o1 if o1[i] == o2.get(i))
        report["determinism"] = {"arm": "qwen-rag-k3", "identical_outputs": same, "n": len(o1)}

    rep_runs = runs.get("qwen-rag-k3-t08", [])
    if len(rep_runs) >= 2:
        outs = [outputs_of(r) for r in rep_runs]
        verdicts = [
            {r.test_case_id: verdict(r, "llm_judge") for r in rr.result.test_case_results}
            for rr in rep_runs
        ]  # type: ignore[union-attr]
        ids = list(outs[0])
        all_same_output = sum(1 for i in ids if len({o.get(i) for o in outs}) == 1)
        judged = [i for i in ids if all(v.get(i) is not None for v in verdicts)]
        unanimous = sum(1 for i in judged if len({v[i] for v in verdicts}) == 1)
        rates = [
            s["evaluators"]["llm_judge"]["pass_rate"] for s in report["arms"]["qwen-rag-k3-t08"]
        ]
        report["consistency"] = {
            "arm": "qwen-rag-k3-t08",
            "repetitions": len(rep_runs),
            "questions": len(ids),
            "all_outputs_identical": all_same_output,
            "judge_verdict_unanimous": unanimous,
            "judge_verdict_flipped": len(judged) - unanimous,
            "judge_pass_rate_per_run": rates,
        }

    args.out.mkdir(parents=True, exist_ok=True)
    write_per_case_csv(runs, args.out / "per_case.csv")
    (args.out / "results.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    (args.out / "results_tables.md").write_text(render_tables(report))
    print(render_tables(report))
    return 0


def write_per_case_csv(runs: dict[str, list[ExperimentRun]], path: Path) -> None:
    """One row per (arm, question) from each arm's first run.

    Compact, human-reviewable record of every answer and verdict (not a raw
    dump: no prompts or retrieved text). The empty ``human_correct`` and
    ``human_notes`` columns make it the annotation sheet for validating the
    LLM judge against human labels (recommended next experiment).
    """
    import csv

    fields = [
        "arm",
        "test_case_id",
        "question",
        "gold_answers",
        "answer",
        "exact_match",
        "squad_em",
        "squad_f1",
        "llm_judge_label",
        "faithfulness_label",
        "retrieved_chunk_ids",
        "relevant_chunk_ids",
        "human_correct",
        "human_notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for arm, arm_runs in runs.items():
            run = arm_runs[0]
            cases = {tc.id: tc for tc in run.dataset_version.dataset.test_cases}
            for r in run.result.test_case_results:  # type: ignore[union-attr]
                tc = cases[r.test_case_id]
                ev = {e.evaluator_name: e for e in r.evaluation_results}
                rag = (
                    r.model_response.provider_metadata.get("rag") if r.model_response else None
                ) or {}
                sm = ev.get("squad_match")
                writer.writerow(
                    {
                        "arm": arm,
                        "test_case_id": r.test_case_id,
                        "question": r.input_text,
                        "gold_answers": " | ".join(gold_answers_for(tc)),
                        "answer": r.model_response.output_text if r.model_response else "",
                        "exact_match": ev["exact_match"].passed if "exact_match" in ev else "",
                        "squad_em": sm.passed if sm else "",
                        "squad_f1": f"{sm.score:.3f}" if sm and sm.score is not None else "",
                        "llm_judge_label": ev["llm_judge"].label if "llm_judge" in ev else "",
                        "faithfulness_label": ev["llm_faithfulness_judge"].label
                        if "llm_faithfulness_judge" in ev
                        else "",
                        "retrieved_chunk_ids": " ".join(
                            c["chunk_id"] for c in rag.get("retrieved_chunks", [])
                        ),
                        "relevant_chunk_ids": " ".join(tc.metadata.get("relevant_chunk_ids", [])),
                        "human_correct": "",
                        "human_notes": "",
                    }
                )


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def render_tables(report: dict[str, Any]) -> str:
    lines = [
        "# EXP-001 results tables (generated)",
        "",
        "Generated by `scripts/analyze_reliability_study.py` from the persisted runs. Do not edit "
        "by hand; interpretation is in `results.md`.",
        "",
        "## Correctness by arm (first run of each arm; 95% Wilson interval)",
        "",
        "| Arm | Exact match | SQuAD EM | SQuAD F1 | LLM judge correct | Abstained |",
        "|---|---|---|---|---|---|",
    ]
    for arm, summaries in report["arms"].items():
        s = summaries[0]
        ev = s["evaluators"]

        def cell(name: str, ev: dict[str, Any] = ev) -> str:
            e = ev.get(name)
            if not e or e["pass_rate"] is None:
                return "n/a"
            lo, hi = e["ci95"]
            return f"{pct(e['pass_rate'])} [{pct(lo)}, {pct(hi)}] (n={e['scored']})"

        f1 = ev.get("squad_match", {}).get("score_mean")
        lines.append(
            f"| {arm} | {cell('exact_match')} | {cell('squad_match')} | "
            f"{'n/a' if f1 is None else f'{f1:.3f}'} | {cell('llm_judge')} | {pct(s['abstention_rate'])} |"
        )
    lines += [
        "",
        "## Faithfulness judge labels (RAG arms, first run)",
        "",
        "| Arm | supported | unsupported | contradicted | undetermined | errors |",
        "|---|---|---|---|---|---|",
    ]
    for arm, summaries in report["arms"].items():
        f = summaries[0]["evaluators"].get("llm_faithfulness_judge")
        if not f or f["status_counts"].get("success", 0) == 0:
            continue  # closed-book arms: every faithfulness check is SKIPPED
        lc, sc = f["label_counts"], f["status_counts"]
        errors = sum(v for k, v in sc.items() if k not in ("success", "skipped"))
        lines.append(
            f"| {arm} | {lc.get('supported', 0)} | {lc.get('unsupported', 0)} | "
            f"{lc.get('contradicted', 0)} | {lc.get('undetermined', 0)} | {errors} |"
        )
    lines += [
        "",
        "## Retrieval (scored at each run's own K)",
        "",
        "| Arm | K | Hit@K strict | Recall@K strict | MRR strict | Hit@K lenient | Judge correct given hit | given miss |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm, summaries in report["arms"].items():
        r = summaries[0].get("retrieval")
        if not r:
            continue
        sm = r["strict_mean"]
        gh, gm = r["judge_correct_given_hit"], r["judge_correct_given_miss"]
        lines.append(
            f"| {arm} | {r['k']} | {pct(sm.get('hit_at_k'))} | {pct(sm.get('recall_at_k'))} | "
            f"{sm.get('mrr', 0):.3f} | {pct(r['lenient_answer_containment_hit_rate'])} | "
            f"{pct(gh[0])} (n={r['n_hit']}) | {pct(gm[0])} (n={r['n_miss']}) |"
        )
    lines += ["", "## Paired contrasts (same questions; exact McNemar test)", ""]
    lines += [
        "| Contrast | Evaluator | n | A pass | B pass | only A | only B | p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in report["contrasts"]:
        for r in c["results"]:
            lines.append(
                f"| {c['label']} | {r['evaluator']} | {r['n_paired']} | {r['a_pass']} | {r['b_pass']} | "
                f"{r['a_only']} | {r['b_only']} | {r['mcnemar_exact_p']:.4f} |"
            )
    lines += ["", "## Evaluator agreement on correctness (first run of each arm)", ""]
    lines += [
        "| Arm | Pair | n | Raw agreement | Cohen's kappa | both pass / A only / B only / both fail |",
        "|---|---|---|---|---|---|",
    ]
    for arm, summaries in report["arms"].items():
        for a in summaries[0]["agreement"]:
            kappa = "undefined" if a["cohens_kappa"] is None else f"{a['cohens_kappa']:.2f}"
            t = a["table_both_pass_a_only_b_only_both_fail"]
            lines.append(
                f"| {arm} | {a['pair']} | {a['cases']} | {pct(a['agreement_rate'])} | {kappa} | {t[0]} / {t[1]} / {t[2]} / {t[3]} |"
            )
    lines += ["", "## Rule-based failure categories (first run)", ""]
    for arm, summaries in report["arms"].items():
        cats = ", ".join(f"{k}: {v}" for k, v in sorted(summaries[0]["failure_categories"].items()))
        lines.append(f"- **{arm}**: {cats}")
    lines += ["", "## Latency and tokens (generation call only, first run)", ""]
    lines += ["| Arm | median ms | p90 ms | max ms | mean total tokens |", "|---|---|---|---|---|"]
    for arm, summaries in report["arms"].items():
        lat = summaries[0]["latency_ms"] or {}
        tok = summaries[0]["total_tokens_mean"]
        lines.append(
            f"| {arm} | {lat.get('median', 0):.0f} | {lat.get('p90', 0):.0f} | {lat.get('max', 0):.0f} | "
            f"{'n/a' if tok is None else f'{tok:.0f}'} |"
        )
    d, c = report["determinism"], report["consistency"]
    lines += ["", "## Determinism and consistency", ""]
    if d:
        lines.append(
            f"- Two T=0 runs of {d['arm']}: {d['identical_outputs']}/{d['n']} normalized outputs identical."
        )
    if c:
        lines.append(
            f"- {c['repetitions']} T=0.8 runs of {c['arm']}: all outputs identical for "
            f"{c['all_outputs_identical']}/{c['questions']} questions; judge verdict flipped on "
            f"{c['judge_verdict_flipped']} questions; judge pass rate per run: "
            + ", ".join(pct(x) for x in c["judge_pass_rate_per_run"])
            + "."
        )
    env = report.get("environment") or {}
    lines += ["", "## Environment", "", "```json", json.dumps(env, indent=1), "```", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
