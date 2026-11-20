"""
EXP-003: re-judge stored answers with judge rubric v1 and v2 (no new generation).

See research/experiments/EXP-003_judge_rubric_v2/definition.md (written before
the run). For each source run, two experiments are created whose dataset and
model configuration equal the source's (so ExperimentManager can verify the
replay), differing only in the judge rubrics:

    exp003-<source>-v1   llm_judge v1, llm_faithfulness_judge v1, squad_match
    exp003-<source>-v2   llm_judge v2, llm_faithfulness_judge v2, squad_match

Each is executed once with ReplayAdapter(source run). Then the analysis is
written to research/experiments/EXP-003_judge_rubric_v2/.

Usage:
    python scripts/run_rejudge_study.py                 # run (resumable) + analyze
    python scripts/run_rejudge_study.py --analyze-only
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_reliability_study as exp001  # noqa: E402
from analyze_reliability_study import fmt_p, mcnemar_exact, verdict  # noqa: E402

from llm_reliability.analytics.agreement import cohens_kappa  # noqa: E402
from llm_reliability.evaluation.llm_judge import (  # noqa: E402
    CORRECTNESS_RUBRIC,
    CORRECTNESS_RUBRIC_V2,
    LLMJudgeEvaluator,
)
from llm_reliability.evaluation.models import (  # noqa: E402
    EvaluationStatus,
    ExecutionRequest,
    ModelConfig,
)
from llm_reliability.evaluation.replay import ReplayAdapter  # noqa: E402
from llm_reliability.evaluation.squad_match import (  # noqa: E402
    SquadNormalizedMatchEvaluator,
    squad_normalize,
)
from llm_reliability.experiments import (  # noqa: E402
    ExperimentManager,
    LocalFileExperimentRepository,
)
from llm_reliability.experiments.models import ExperimentRun  # noqa: E402
from llm_reliability.rag.faithfulness_judge import (  # noqa: E402
    FAITHFULNESS_RUBRIC,
    FAITHFULNESS_RUBRIC_V2,
    LLMFaithfulnessEvaluator,
)

OUT_DIR = exp001.REPO_ROOT / "research" / "experiments" / "EXP-003_judge_rubric_v2"
SOURCES = {
    "qwen-rag-k3": "exp001-qwen-rag-k3",
    "gemma-rag-k3": "exp001-gemma-rag-k3",
    "qwen-closed": "exp001-qwen-closed",
    "freeform": "exp002-freeform",
}
RUBRICS = {
    "v1": (CORRECTNESS_RUBRIC, FAITHFULNESS_RUBRIC),
    "v2": (CORRECTNESS_RUBRIC_V2, FAITHFULNESS_RUBRIC_V2),
}
ABSTAIN = "i dont know"


def judge_config() -> ModelConfig:
    return ModelConfig(
        model_id=exp001.JUDGE,
        temperature=0.0,
        max_tokens=200,
        extra_params={"seed": exp001.SEED, "response_format": {"type": "json_object"}},
    )


def evaluators(version: str) -> list[Any]:
    correctness, faithfulness = RUBRICS[version]
    judge = exp001.provider(exp001.JUDGE)
    return [
        SquadNormalizedMatchEvaluator(),
        LLMJudgeEvaluator(judge, judge_config(), rubric=correctness),
        LLMFaithfulnessEvaluator(judge, judge_config(), rubric=faithfulness),
    ]


def first_run(repo: LocalFileExperimentRepository, exp_id: str) -> ExperimentRun:
    runs = sorted(
        (repo.load_run(r) for r in repo.list_run_ids(exp_id)), key=lambda r: r.repetition_index
    )
    if not runs:
        raise SystemExit(f"No runs for {exp_id}. Restore EXP-001 runs.tar.gz / run EXP-002 first.")
    return runs[0]


def label(tc: Any, evaluator: str) -> str | None:
    for e in tc.evaluation_results:
        if e.evaluator_name == evaluator:
            return e.label if e.status == EvaluationStatus.SUCCESS else f"<{e.status.value}>"
    return None


def run_all(repo: LocalFileExperimentRepository) -> None:
    manager = ExperimentManager(repo)
    exp001.provider(exp001.JUDGE).generate(
        ExecutionRequest("warmup", "Say OK.", ModelConfig(exp001.JUDGE, 0.0, 4))
    )
    for key, source_exp in SOURCES.items():
        source = first_run(repo, source_exp)
        for version in RUBRICS:
            exp_id = f"exp003-{key}-{version}"
            evs = evaluators(version)
            if exp_id not in repo.list_experiment_ids():
                manager.create_experiment(
                    experiment_id=exp_id,
                    name=f"EXP-003 · re-judge {source_exp} · rubric {version}",
                    dataset=source.dataset_version.dataset,
                    model_config=source.model_config,
                    evaluators=evs,
                    description=(
                        f"EXP-003: the recorded answers of run {source.run_id} ({source_exp}) "
                        f"re-judged with rubric {version}; no new generation (ReplayAdapter)."
                    ),
                    metadata={
                        "study": "EXP-003",
                        "source_run_id": source.run_id,
                        "rubric": version,
                    },
                )
            if repo.list_run_ids(exp_id):
                continue
            print(
                f"[{exp_id}] judging {len(source.dataset_version.dataset.test_cases)} answers",
                flush=True,
            )
            manager.execute_experiment(
                manager.load_experiment(exp_id), ReplayAdapter(source), evs, adapter_id="replay"
            )


def analyze(repo: LocalFileExperimentRepository) -> dict[str, Any]:
    report: dict[str, Any] = {"study": "EXP-003", "sources": {}}
    for key, source_exp in SOURCES.items():
        original = first_run(repo, source_exp)
        v1 = first_run(repo, f"exp003-{key}-v1")
        v2 = first_run(repo, f"exp003-{key}-v2")
        o = {r.test_case_id: r for r in original.result.test_case_results}  # type: ignore[union-attr]
        a = {r.test_case_id: r for r in v1.result.test_case_results}  # type: ignore[union-attr]
        b = {r.test_case_id: r for r in v2.result.test_case_results}  # type: ignore[union-attr]
        ids = [i for i in o if i in a and i in b and o[i].model_response]
        abst = {i for i in ids if squad_normalize(o[i].model_response.output_text) == ABSTAIN}
        gold = {
            tc.id: tc.metadata.get("gold_answers") or [tc.reference_answer]
            for tc in original.dataset_version.dataset.test_cases
        }

        def flips(
            x: dict[str, Any], y: dict[str, Any], ev: str, ids: list[str] = ids
        ) -> dict[str, Any]:
            pairs = [(verdict(x[i], ev), verdict(y[i], ev)) for i in ids]
            scored = [(p, q) for p, q in pairs if p is not None and q is not None]
            pf = sum(1 for p, q in scored if p and not q)
            fp = sum(1 for p, q in scored if not p and q)
            return {
                "n": len(scored),
                "pass_before": sum(1 for p, _ in scored if p),
                "pass_after": sum(1 for _, q in scored if q),
                "pass_to_fail": pf,
                "fail_to_pass": fp,
                "mcnemar_exact_p": mcnemar_exact(pf, fp),
            }

        def kappa_vs_squad(runs: dict[str, Any], ids: list[str] = ids) -> dict[str, Any]:
            both = [(verdict(runs[i], "llm_judge"), verdict(runs[i], "squad_match")) for i in ids]
            both = [(j, s) for j, s in both if j is not None and s is not None]
            table = (
                sum(1 for j, s in both if j and s),
                sum(1 for j, s in both if j and not s),
                sum(1 for j, s in both if not j and s),
                sum(1 for j, s in both if not j and not s),
            )
            return {
                "n": len(both),
                "agreement": (table[0] + table[3]) / len(both) if both else None,
                "kappa": cohens_kappa(
                    {
                        (True, True): table[0],
                        (True, False): table[1],
                        (False, True): table[2],
                        (False, False): table[3],
                    }
                )
                if both
                else None,
                "table_both_pass_judge_only_squad_only_both_fail": table,
            }

        def faith_by_abstention(
            runs: dict[str, Any], ids: list[str] = ids, abst: set[str] = abst
        ) -> dict[str, dict[str, int]]:
            out: dict[str, Counter[str]] = {"abstention": Counter(), "answer": Counter()}
            for i in ids:
                lab = label(runs[i], "llm_faithfulness_judge")
                if lab is not None:
                    out["abstention" if i in abst else "answer"][lab] += 1
            return {k: dict(v) for k, v in out.items()}

        def failures(runs: dict[str, Any], ids: list[str] = ids) -> dict[str, int]:
            c: Counter[str] = Counter()
            for i in ids:
                for e in runs[i].evaluation_results:
                    if e.status in (
                        EvaluationStatus.OUTPUT_VALIDATION_ERROR,
                        EvaluationStatus.EXECUTION_ERROR,
                    ):
                        c[f"{e.evaluator_name}:{e.status.value}"] += 1
            return dict(c)

        changed = []
        for i in ids:
            lv1, lv2 = label(a[i], "llm_judge"), label(b[i], "llm_judge")
            if verdict(a[i], "llm_judge") != verdict(b[i], "llm_judge"):
                changed.append(
                    {
                        "test_case_id": i,
                        "question": o[i].input_text,
                        "answer": o[i].model_response.output_text,
                        "gold": gold.get(i),
                        "v1": lv1,
                        "v2": lv2,
                        "squad_em": verdict(a[i], "squad_match"),
                    }
                )
        report["sources"][key] = {
            "source_run": original.run_id,
            "rejudge_v1_run": v1.run_id,
            "v2_run": v2.run_id,
            "n": len(ids),
            "abstentions": len(abst),
            "judge_noise_correctness (original v1 vs re-judged v1)": flips(o, a, "llm_judge"),
            "judge_noise_faithfulness (original v1 vs re-judged v1)": flips(
                o, a, "llm_faithfulness_judge"
            ),
            "rubric_effect_correctness (v1 vs v2)": flips(a, b, "llm_judge"),
            "rubric_effect_faithfulness (v1 vs v2)": flips(a, b, "llm_faithfulness_judge"),
            "correctness_labels": {
                "v1": dict(Counter(label(a[i], "llm_judge") for i in ids)),
                "v2": dict(Counter(label(b[i], "llm_judge") for i in ids)),
            },
            "faithfulness_by_abstention": {
                "v1": faith_by_abstention(a),
                "v2": faith_by_abstention(b),
            },
            "judge_vs_squad_em": {"v1": kappa_vs_squad(a), "v2": kappa_vs_squad(b)},
            "judge_failures": {"v1": failures(a), "v2": failures(b)},
            "correctness_changes_v1_to_v2": changed,
        }
    return report


def render(report: dict[str, Any]) -> str:
    lines = [
        "# EXP-003 result tables (generated by scripts/run_rejudge_study.py)",
        "",
        "Same recorded answers in every column; only the judge rubric (or a repeat of it) differs.",
        "",
        "## Correctness judge",
        "",
        "| Source | n | Judge noise: flips v1 orig → v1 again | v1 pass → v2 pass | v1→v2 flips (→fail / →pass) | McNemar p | κ vs SQuAD EM v1 → v2 |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, s in report["sources"].items():
        noise = s["judge_noise_correctness (original v1 vs re-judged v1)"]
        eff = s["rubric_effect_correctness (v1 vs v2)"]
        k1, k2 = s["judge_vs_squad_em"]["v1"]["kappa"], s["judge_vs_squad_em"]["v2"]["kappa"]
        lines.append(
            f"| {key} | {s['n']} | {noise['pass_to_fail'] + noise['fail_to_pass']} | "
            f"{eff['pass_before']} → {eff['pass_after']} | {eff['pass_to_fail']} / {eff['fail_to_pass']} | "
            f"{fmt_p(eff['mcnemar_exact_p'])} | "
            f"{'n/a' if k1 is None else f'{k1:.2f}'} → {'n/a' if k2 is None else f'{k2:.2f}'} |"
        )
    lines += [
        "",
        "## Faithfulness judge: labels for abstentions vs answers",
        "",
        "| Source | Abstentions | v1 labels on abstentions | v2 labels on abstentions | v2 `no_claim` on non-abstentions | Judge noise flips (v1) |",
        "|---|---|---|---|---|---|",
    ]
    for key, s in report["sources"].items():
        f1, f2 = s["faithfulness_by_abstention"]["v1"], s["faithfulness_by_abstention"]["v2"]
        noise = s["judge_noise_faithfulness (original v1 vs re-judged v1)"]
        lines.append(
            f"| {key} | {s['abstentions']} | {f1['abstention'] or '—'} | {f2['abstention'] or '—'} | "
            f"{f2['answer'].get('no_claim', 0)} of {sum(f2['answer'].values())} | "
            f"{noise['pass_to_fail'] + noise['fail_to_pass']} |"
        )
    lines += ["", "## Judge failures (not scored)", ""]
    for key, s in report["sources"].items():
        lines.append(
            f"- {key}: v1 {s['judge_failures']['v1'] or 'none'}; v2 {s['judge_failures']['v2'] or 'none'}"
        )
    lines += ["", "## Every correctness verdict that changed between v1 and v2", ""]
    lines += ["| Source | Question | Answer | v1 | v2 | SQuAD EM |", "|---|---|---|---|---|---|"]
    for key, s in report["sources"].items():
        for c in s["correctness_changes_v1_to_v2"]:
            ans = c["answer"].replace("|", "/").replace("\n", " ")[:90]
            lines.append(
                f"| {key} | {c['question'][:70]} | {ans} | {c['v1']} | {c['v2']} | "
                f"{'pass' if c['squad_em'] else 'fail'} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=exp001.REPO_ROOT / "experiments")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--sources", default=",".join(SOURCES), help="subset of source keys")
    args = parser.parse_args()
    for key in set(SOURCES) - set(args.sources.split(",")):
        del SOURCES[key]
    repo = LocalFileExperimentRepository(args.store)
    if not args.analyze_only:
        run_all(repo)
    report = analyze(repo)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (OUT_DIR / "results_tables.md").write_text(render(report))
    print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
