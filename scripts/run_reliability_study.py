"""
EXP-001: run the reliability study against local models served by Ollama.

Design (see research/experiments/EXP-001_local_rag_reliability/definition.md):

    arm                 generator           retrieval      temperature  runs
    qwen-closed         qwen3:4b-instruct   none           0            1
    qwen-rag-k1         qwen3:4b-instruct   top-1          0            1
    qwen-rag-k3         qwen3:4b-instruct   top-3          0            2  (determinism check)
    gemma-closed        gemma3:1b           none           0            1
    gemma-rag-k3        gemma3:1b           top-3          0            1
    qwen-rag-k3-t08     qwen3:4b-instruct   top-3          0.8          3  (consistency)

Every arm uses the same dataset version (squad-v1.1-dev-subset-v1 with
relevant_chunk_ids derived for the chunking below), the same answer-format
instruction, and the same four evaluators:

    exact_match            project baseline (case/whitespace only, first gold answer)
    squad_match            SQuAD official normalization, EM over all gold answers, F1 score
    llm_judge              correctness vs reference, judged by qwen3:4b-instruct
    llm_faithfulness_judge support by retrieved context, judged by qwen3:4b-instruct
                           (SKIPPED automatically for closed-book arms)

Runs go through ExperimentManager into a normal experiment store, so they
appear in the API/UI like any other run and carry full provenance
(adapter_id, executed model ids, config fingerprint, environment record).

Usage:
    python scripts/run_reliability_study.py                      # full study -> ./experiments
    python scripts/run_reliability_study.py --limit 5 --store /tmp/pilot --arms qwen-rag-k3

Requires a running Ollama server with the three models pulled. The script is
resumable: experiments/runs already present in the store are skipped.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_reliability.datasets.squad import SQUAD_CHUNKING, SQUAD_SUBSET_PATH, load_squad_subset
from llm_reliability.evaluation.adapters import ModelAdapter
from llm_reliability.evaluation.evaluators import Evaluator, ExactMatchEvaluator
from llm_reliability.evaluation.llm_judge import LLMJudgeEvaluator
from llm_reliability.evaluation.models import (
    ExecutionRequest,
    ModelConfig,
    ModelResponse,
)
from llm_reliability.evaluation.prompting import PromptTemplateAdapter
from llm_reliability.evaluation.squad_match import SquadNormalizedMatchEvaluator
from llm_reliability.experiments import ExperimentManager, LocalFileExperimentRepository
from llm_reliability.providers.base import ProviderConfig
from llm_reliability.providers.openai_provider import OpenAIProviderAdapter
from llm_reliability.rag.embeddings import EmbeddingModel, OpenAICompatibleEmbeddingModel
from llm_reliability.rag.faithfulness_judge import LLMFaithfulnessEvaluator
from llm_reliability.rag.pipeline import RagPipeline, RagPipelineConfig
from llm_reliability.rag.retriever import Retriever

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = SQUAD_SUBSET_PATH
OLLAMA = "http://localhost:11434"
BASE_URL = f"{OLLAMA}/v1"

QWEN = "qwen3:4b-instruct"
GEMMA = "gemma3:1b"
JUDGE = QWEN
EMBEDDING = "nomic-embed-text"
CHUNKING = SQUAD_CHUNKING
SEED = 42
MAX_TOKENS = 64
CLOSED_BOOK_TEMPLATE = "closed_book_concise_v1"
RAG_TEMPLATE = "concise_v1"
STUDY_ID = "EXP-001"


@dataclass(frozen=True)
class Arm:
    key: str
    generator: str
    top_k: int | None  # None = closed book
    temperature: float
    runs: int
    seed: int | None = SEED


ARMS = [
    Arm("qwen-closed", QWEN, None, 0.0, 1),
    Arm("qwen-rag-k1", QWEN, 1, 0.0, 1),
    Arm("qwen-rag-k3", QWEN, 3, 0.0, 2),
    Arm("gemma-closed", GEMMA, None, 0.0, 1),
    Arm("gemma-rag-k3", GEMMA, 3, 0.0, 1),
    # No fixed seed: with a fixed seed, sampling at T=0.8 would repeat itself
    # and the arm could not measure run-to-run variation.
    Arm("qwen-rag-k3-t08", QWEN, 3, 0.8, 3, seed=None),
]


class CachingEmbeddingModel(EmbeddingModel):
    """Embeds each distinct text once (the corpus is shared by every RAG arm)."""

    def __init__(self, inner: OpenAICompatibleEmbeddingModel) -> None:
        self._inner = inner
        self.model_id = inner.model_id
        self._docs: dict[str, list[float]] = {}
        self._queries: dict[str, list[float]] = {}

    def get_config(self) -> dict[str, Any]:
        return self._inner.get_config()

    def embed(self, texts):  # type: ignore[no-untyped-def]
        missing = [t for t in dict.fromkeys(texts) if t not in self._docs]
        if missing:
            self._docs.update(zip(missing, self._inner.embed(missing), strict=True))
        return [self._docs[t] for t in texts]

    def embed_one(self, text: str) -> list[float]:
        if text not in self._queries:
            self._queries[text] = self._inner.embed_one(text)
        return self._queries[text]


class ProgressAdapter(ModelAdapter):
    """Prints a progress line every few test cases; passes everything through."""

    def __init__(self, inner: ModelAdapter, label: str, total: int) -> None:
        self._inner, self._label, self._total, self._done = inner, label, total, 0
        self._start = time.monotonic()

    def __getattr__(self, name: str) -> Any:  # expose inner .config for RAG checks
        return getattr(self._inner, name)

    def generate(self, request: ExecutionRequest) -> ModelResponse:
        try:
            return self._inner.generate(request)
        finally:
            self._done += 1
            if self._done % 10 == 0 or self._done == self._total:
                elapsed = time.monotonic() - self._start
                print(f"    {self._label}: {self._done}/{self._total} ({elapsed:.0f}s)", flush=True)


def _get_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310 - fixed local URL
        return json.loads(resp.read())


def environment_record() -> dict[str, Any]:
    tags = _get_json(f"{OLLAMA}/api/tags")
    digests = {m["name"]: m["digest"] for m in tags.get("models", [])}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--", "src", "scripts"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except OSError:
        commit, dirty = "unknown", True
    return {
        "study_id": STUDY_ID,
        "ollama_version": _get_json(f"{OLLAMA}/api/version").get("version"),
        "model_digests": {
            name: digests.get(name) or digests.get(f"{name}:latest")
            for name in (QWEN, GEMMA, EMBEDDING)
        },
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": commit,
        "git_dirty_src_or_scripts": dirty,
    }


def provider(model_id: str) -> OpenAIProviderAdapter:
    return OpenAIProviderAdapter(
        config=ProviderConfig(
            provider_id="ollama-openai-compatible",
            model_id=model_id,
            base_url=BASE_URL,
            timeout_seconds=180.0,
            max_retries=2,
            allow_external_inference=True,
        )
    )


def evaluators() -> list[Evaluator]:
    judge_config = ModelConfig(
        model_id=JUDGE,
        temperature=0.0,
        max_tokens=200,
        extra_params={"seed": SEED, "response_format": {"type": "json_object"}},
    )
    return [
        ExactMatchEvaluator(),
        SquadNormalizedMatchEvaluator(),
        LLMJudgeEvaluator(judge_adapter=provider(JUDGE), judge_model_config=judge_config),
        LLMFaithfulnessEvaluator(judge_adapter=provider(JUDGE), judge_model_config=judge_config),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--store", type=Path, default=REPO_ROOT / "experiments")
    parser.add_argument("--limit", type=int, default=None, help="pilot: first N questions only")
    parser.add_argument("--arms", default=",".join(a.key for a in ARMS))
    parser.add_argument(
        "--allow-env-change",
        action="store_true",
        help="append runs to existing experiments even if Ollama model digests differ",
    )
    args = parser.parse_args()

    subset = load_squad_subset(DATA_PATH)
    documents = subset.documents()
    chunks = subset.chunks(CHUNKING)
    dataset = subset.dataset(CHUNKING, args.limit)
    corpus_fingerprint = subset.corpus_fingerprint()
    env = environment_record()
    print(f"{len(dataset.test_cases)} questions, {len(documents)} documents, {len(chunks)} chunks")
    print(json.dumps(env, indent=1))

    embedder = CachingEmbeddingModel(
        OpenAICompatibleEmbeddingModel(
            EMBEDDING,
            base_url=BASE_URL,
            document_prefix="search_document: ",
            query_prefix="search_query: ",
        )
    )
    # Warm-up: load every model once before any recorded call, so the first
    # recorded latency of each run does not include model loading (~2-4 s
    # observed in the pilot vs ~0.1-0.5 s per warm call).
    for model in (QWEN, GEMMA):
        provider(model).generate(ExecutionRequest("warmup", "Say OK.", ModelConfig(model, 0.0, 4)))
    embedder.embed_one("warmup")

    repo = LocalFileExperimentRepository(args.store)
    manager = ExperimentManager(repo)
    selected = [a for a in ARMS if a.key in set(args.arms.split(","))]
    suffix = f"-pilot{args.limit}" if args.limit else ""

    for arm in selected:
        exp_id = f"exp001-{arm.key}{suffix}"
        extra: dict[str, Any] = {"seed": arm.seed} if arm.seed is not None else {}
        if arm.top_k is None:
            extra["prompt_template_id"] = CLOSED_BOOK_TEMPLATE
            adapter: ModelAdapter = PromptTemplateAdapter(
                provider(arm.generator), CLOSED_BOOK_TEMPLATE
            )
            adapter_id = "ollama-closed-book"
        else:
            rag_config = RagPipelineConfig(
                embedding_model_id=EMBEDDING,
                chunk_size=CHUNKING.chunk_size,
                chunk_overlap=CHUNKING.chunk_overlap,
                top_k=arm.top_k,
                corpus_fingerprint=corpus_fingerprint,
                prompt_template_id=RAG_TEMPLATE,
            )
            extra["rag"] = rag_config.to_dict()
            adapter = RagPipeline(
                retriever=Retriever(chunks=chunks, embedding_model=embedder, top_k=arm.top_k),
                generation_adapter=provider(arm.generator),
                config=rag_config,
            )
            adapter_id = f"ollama-rag-k{arm.top_k}"
        model_config = ModelConfig(
            model_id=arm.generator,
            temperature=arm.temperature,
            max_tokens=MAX_TOKENS,
            extra_params=extra,
        )
        evs = evaluators()
        if exp_id in repo.list_experiment_ids():
            definition = manager.load_experiment(exp_id)
            recorded = (definition.metadata.get("study") or {}).get("model_digests")
            if recorded and recorded != env["model_digests"] and not args.allow_env_change:
                raise SystemExit(
                    f"{exp_id} was defined with model digests {recorded}, but this machine has "
                    f"{env['model_digests']}. New runs would be filed under an environment that "
                    "did "
                    "not produce them. Use a new --store (fresh experiments), or pass "
                    "--allow-env-change to append anyway (and document it)."
                )
        else:
            retrieval = "closed book" if arm.top_k is None else f"top-{arm.top_k} retrieval"
            definition = manager.create_experiment(
                experiment_id=exp_id,
                name=f"EXP-001 · {arm.generator} · {retrieval} · T={arm.temperature}",
                dataset=dataset,
                model_config=model_config,
                evaluators=evs,
                description=(
                    f"{STUDY_ID} arm '{arm.key}': {len(dataset.test_cases)} SQuAD v1.1 questions "
                    f"answered by {arm.generator} ({retrieval}, temperature {arm.temperature}); "
                    f"judged by {JUDGE}."
                ),
                metadata={"study": env, "arm": arm.key},
            )
        existing = len(repo.list_run_ids(exp_id))
        if existing < arm.runs:
            # Re-warm before each arm: Ollama unloads models idle for 5 minutes
            # (keep_alive), so a start-up warm-up alone left a cold load in the
            # first recorded call of later arms (seen as one max-latency outlier
            # in the original EXP-001 gemma arms).
            for model in {arm.generator, JUDGE}:
                provider(model).generate(
                    ExecutionRequest("warmup", "Say OK.", ModelConfig(model, 0.0, 4))
                )
        for rep in range(existing + 1, arm.runs + 1):
            print(f"[{exp_id}] run {rep}/{arm.runs}", flush=True)
            run = manager.execute_experiment(
                definition,
                ProgressAdapter(adapter, arm.key, len(dataset.test_cases)),
                evs,
                adapter_id=adapter_id,
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
