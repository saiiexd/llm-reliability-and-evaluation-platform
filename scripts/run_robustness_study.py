"""
EXP-002: robustness of the local RAG pipeline to controlled perturbations (RQ5).

Design (see research/experiments/EXP-002_robustness/definition.md, written
before the run):

    arm         change vs baseline                                runs
    base        none (qwen3:4b-instruct, top-3, concise_v1, T=0)  2  (noise floor)
    reversed    context_order = "reversed"                        1
    distractor  distractor_chunks = 1 (least-similar chunk first) 1
    paraphrase  questions replaced by their frozen paraphrases    1
    freeform    prompt_template_id = "default_v1"                 1

Everything else is shared with EXP-001 (same helpers are imported): judge,
evaluators, embedding model and prefixes, chunking, warm-up, provenance.
Runs go through ExperimentManager into the normal store. Resumable.

Usage:
    python scripts/run_robustness_study.py                       # full study -> ./experiments
    python scripts/run_robustness_study.py --limit 3 --store /tmp/pilot --arms base
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_reliability_study as exp001  # noqa: E402

from llm_reliability.datasets.squad import (  # noqa: E402
    SQUAD_CHUNKING,
    load_squad_subset,
    paraphrased_dataset,
)
from llm_reliability.evaluation.models import (  # noqa: E402
    Dataset,
    ExecutionRequest,
    ModelConfig,
)
from llm_reliability.experiments import (  # noqa: E402
    ExperimentManager,
    LocalFileExperimentRepository,
)
from llm_reliability.rag.embeddings import OpenAICompatibleEmbeddingModel  # noqa: E402
from llm_reliability.rag.pipeline import RagPipeline, RagPipelineConfig  # noqa: E402
from llm_reliability.rag.retriever import Retriever  # noqa: E402

STUDY_ID = "EXP-002"
GENERATOR = exp001.QWEN
TOP_K = 3


@dataclass(frozen=True)
class Arm:
    key: str
    runs: int
    context_order: str = "ranked"
    distractor_chunks: int = 0
    template: str = exp001.RAG_TEMPLATE
    paraphrased: bool = False
    description: str = ""


ARMS = [
    Arm("base", 2, description="baseline (no change)"),
    Arm("reversed", 1, context_order="reversed", description="passages in reversed order"),
    Arm("distractor", 1, distractor_chunks=1, description="least-similar passage added first"),
    Arm("paraphrase", 1, paraphrased=True, description="questions reworded"),
    Arm("freeform", 1, template="default_v1", description="free-form answer template"),
]


def _limit(dataset: Dataset, limit: int | None) -> Dataset:
    if not limit:
        return dataset
    return Dataset(name=f"{dataset.name}-first{limit}", test_cases=dataset.test_cases[:limit])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--store", type=Path, default=exp001.REPO_ROOT / "experiments")
    parser.add_argument("--limit", type=int, default=None, help="pilot: first N questions only")
    parser.add_argument("--arms", default=",".join(a.key for a in ARMS))
    args = parser.parse_args()

    subset = load_squad_subset()
    chunks = subset.chunks(SQUAD_CHUNKING)
    corpus_fingerprint = subset.corpus_fingerprint()
    original = subset.dataset(SQUAD_CHUNKING, args.limit)
    paraphrased = _limit(paraphrased_dataset(SQUAD_CHUNKING), args.limit)
    env = exp001.environment_record()
    env["study_id"] = STUDY_ID
    print(f"{len(original.test_cases)} questions, {len(chunks)} chunks")

    embedder = exp001.CachingEmbeddingModel(
        OpenAICompatibleEmbeddingModel(
            exp001.EMBEDDING,
            base_url=exp001.BASE_URL,
            document_prefix="search_document: ",
            query_prefix="search_query: ",
        )
    )
    repo = LocalFileExperimentRepository(args.store)
    manager = ExperimentManager(repo)
    selected = [a for a in ARMS if a.key in set(args.arms.split(","))]
    suffix = f"-pilot{args.limit}" if args.limit else ""

    for arm in selected:
        exp_id = f"exp002-{arm.key}{suffix}"
        dataset = paraphrased if arm.paraphrased else original
        rag_config = RagPipelineConfig(
            embedding_model_id=exp001.EMBEDDING,
            chunk_size=SQUAD_CHUNKING.chunk_size,
            chunk_overlap=SQUAD_CHUNKING.chunk_overlap,
            top_k=TOP_K,
            corpus_fingerprint=corpus_fingerprint,
            prompt_template_id=arm.template,
            context_order=arm.context_order,
            distractor_chunks=arm.distractor_chunks,
        )
        adapter = RagPipeline(
            retriever=Retriever(chunks=chunks, embedding_model=embedder, top_k=TOP_K),
            generation_adapter=exp001.provider(GENERATOR),
            config=rag_config,
        )
        model_config = ModelConfig(
            model_id=GENERATOR,
            temperature=0.0,
            max_tokens=exp001.MAX_TOKENS,
            extra_params={"seed": exp001.SEED, "rag": rag_config.to_dict()},
        )
        evs = exp001.evaluators()
        if exp_id in repo.list_experiment_ids():
            definition = manager.load_experiment(exp_id)
            recorded: Any = (definition.metadata.get("study") or {}).get("model_digests")
            if recorded and recorded != env["model_digests"]:
                raise SystemExit(f"{exp_id}: model digests changed since it was defined.")
        else:
            definition = manager.create_experiment(
                experiment_id=exp_id,
                name=f"EXP-002 · {GENERATOR} · top-3 · {arm.description}",
                dataset=dataset,
                model_config=model_config,
                evaluators=evs,
                description=(
                    f"{STUDY_ID} arm '{arm.key}': {arm.description}. "
                    f"{len(dataset.test_cases)} SQuAD v1.1 questions, {GENERATOR}, top-3 "
                    f"retrieval, T=0; judged by {exp001.JUDGE}."
                ),
                metadata={"study": env, "arm": arm.key},
            )
        existing = len(repo.list_run_ids(exp_id))
        if existing < arm.runs:
            for model in {GENERATOR, exp001.JUDGE}:
                exp001.provider(model).generate(
                    ExecutionRequest("warmup", "Say OK.", ModelConfig(model, 0.0, 4))
                )
            embedder.embed_one("warmup")
        for rep in range(existing + 1, arm.runs + 1):
            print(f"[{exp_id}] run {rep}/{arm.runs}", flush=True)
            run = manager.execute_experiment(
                definition,
                exp001.ProgressAdapter(adapter, arm.key, len(dataset.test_cases)),
                evs,
                adapter_id=f"ollama-rag-k3-{arm.key}",
                repetition_index=rep,
            )
            r = run.result
            print(
                f"  -> {run.status.value}: {r.num_succeeded if r else 0} answered, "
                f"run_id={run.run_id}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
