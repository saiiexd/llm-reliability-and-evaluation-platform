"""
Real-model components for the API: OpenAI-compatible generation, retrieval
over the bundled SQuAD corpus, and LLM judges.

Everything here makes outbound requests to the server configured by
``LLM_RELIABILITY_OPENAI_BASE_URL`` (for example a local Ollama server at
``http://localhost:11434/v1``). Callers must have checked the request's
``allow_external_inference`` opt-in first; the factories also refuse to build
without it, so a missed check cannot silently produce network traffic.

Reconstruction, not re-configuration: a live component is always built from
the configuration the experiment *declared* (its ``model_config`` or an
``EvaluatorConfig``), and only when that configuration is one this code can
reproduce exactly (known corpus fingerprint, known rubric). Otherwise a
``ValueError`` explains why, and the run is refused before anything is
recorded.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

from llm_reliability.datasets.squad import SQUAD_CHUNKING, SquadSubset, load_squad_subset
from llm_reliability.evaluation.adapters import ModelAdapter
from llm_reliability.evaluation.llm_judge import (
    CORRECTNESS_RUBRIC,
    CORRECTNESS_RUBRIC_V2,
    JudgeRubric,
    LLMJudgeEvaluator,
)
from llm_reliability.evaluation.models import ExecutionRequest, ModelConfig
from llm_reliability.evaluation.prompting import PromptTemplateAdapter
from llm_reliability.providers.base import ProviderConfig
from llm_reliability.providers.openai_provider import (
    OpenAIProviderAdapter,
    get_openai_base_url_from_env,
)
from llm_reliability.rag.documents import ChunkingConfig
from llm_reliability.rag.embeddings import EmbeddingModel, OpenAICompatibleEmbeddingModel
from llm_reliability.rag.faithfulness_judge import (
    FAITHFULNESS_RUBRIC,
    FAITHFULNESS_RUBRIC_V2,
    LLMFaithfulnessEvaluator,
)
from llm_reliability.rag.pipeline import RagPipeline, RagPipelineConfig
from llm_reliability.rag.retriever import Retriever

# Task prefixes some embedding models are trained with. Part of how vectors are
# computed, so they must be the same for every run that declares this model.
EMBEDDING_PREFIXES: dict[str, tuple[str, str]] = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
}
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
RAG_PROMPT_TEMPLATE = "concise_v1"
CLOSED_BOOK_TEMPLATE = "closed_book_concise_v1"

# The judge settings EXP-001 used, so its experiments can be re-run from the API.
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 200
JUDGE_SEED = 42

GENERATION_TIMEOUT_SECONDS = 180.0
GENERATION_MAX_RETRIES = 2

LIVE_JUDGE_RUBRICS: dict[str, JudgeRubric] = {
    "llm_judge": CORRECTNESS_RUBRIC,
    "llm_faithfulness_judge": FAITHFULNESS_RUBRIC,
}
# Every rubric version a declared judge may use; v1 prompts are kept unchanged
# so experiments recorded with them stay reproducible.
JUDGE_RUBRIC_VERSIONS: dict[str, dict[str, JudgeRubric]] = {
    "llm_judge": {"v1": CORRECTNESS_RUBRIC, "v2": CORRECTNESS_RUBRIC_V2},
    "llm_faithfulness_judge": {"v1": FAITHFULNESS_RUBRIC, "v2": FAITHFULNESS_RUBRIC_V2},
}


def _require_opt_in(allow_external_inference: bool, what: str) -> None:
    if not allow_external_inference:
        raise ValueError(
            f"{what} calls a model server. Tick the external-inference opt-in "
            "(allow_external_inference=true) to permit outbound requests."
        )


def _require_base_url() -> str:
    base_url = get_openai_base_url_from_env()
    if not base_url:
        raise ValueError(
            "No OpenAI-compatible server is configured. Set LLM_RELIABILITY_OPENAI_BASE_URL "
            "(for example http://localhost:11434/v1 for Ollama) in the backend's environment."
        )
    return base_url


def openai_compatible_provider(
    model_id: str, allow_external_inference: bool, client: Any | None = None
) -> OpenAIProviderAdapter:
    """Generation adapter that sends the request's model id to the configured server."""
    return OpenAIProviderAdapter(
        use_request_model=True,
        config=ProviderConfig(
            provider_id="openai-compatible",
            model_id=model_id,
            base_url=get_openai_base_url_from_env(),
            timeout_seconds=GENERATION_TIMEOUT_SECONDS,
            max_retries=GENERATION_MAX_RETRIES,
            allow_external_inference=allow_external_inference,
        ),
        client=client,
    )


# --- Corpora -----------------------------------------------------------------


def known_corpora() -> dict[str, SquadSubset]:
    """Corpora a live RAG pipeline can be rebuilt from, keyed by corpus fingerprint."""
    subset = load_squad_subset()
    return {subset.corpus_fingerprint(): subset}


class _CachingEmbeddingModel(EmbeddingModel):
    """Embeds each distinct passage and query once per backend process.

    Building a retriever embeds the whole corpus; without this, every run of a
    live RAG experiment would re-embed all chunks. Thread-safe because
    background jobs and request handlers can share one instance.
    """

    def __init__(self, inner: OpenAICompatibleEmbeddingModel) -> None:
        self._inner = inner
        self.model_id = inner.model_id
        self._docs: dict[str, list[float]] = {}
        self._queries: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def get_config(self) -> dict[str, Any]:
        return self._inner.get_config()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        with self._lock:
            missing = [t for t in dict.fromkeys(texts) if t not in self._docs]
        if missing:
            vectors = self._inner.embed(missing)
            with self._lock:
                self._docs.update(zip(missing, vectors, strict=True))
        with self._lock:
            return [self._docs[t] for t in texts]

    def embed_one(self, text: str) -> list[float]:
        with self._lock:
            cached = self._queries.get(text)
        if cached is not None:
            return cached
        vector = self._inner.embed_one(text)
        with self._lock:
            self._queries[text] = vector
        return vector


_EMBEDDERS: dict[tuple[str, str], _CachingEmbeddingModel] = {}
_EMBEDDERS_LOCK = threading.Lock()


def cached_embedder(
    embedding_model_id: str, allow_external_inference: bool, client: Any | None = None
) -> EmbeddingModel:
    _require_opt_in(allow_external_inference, f"Embedding model '{embedding_model_id}'")
    base_url = _require_base_url()
    key = (base_url, embedding_model_id)
    with _EMBEDDERS_LOCK:
        if key not in _EMBEDDERS or client is not None:
            doc_prefix, query_prefix = EMBEDDING_PREFIXES.get(embedding_model_id, ("", ""))
            _EMBEDDERS[key] = _CachingEmbeddingModel(
                OpenAICompatibleEmbeddingModel(
                    embedding_model_id,
                    base_url=base_url,
                    client=client,
                    document_prefix=doc_prefix,
                    query_prefix=query_prefix,
                )
            )
        return _EMBEDDERS[key]


def squad_rag_config(
    top_k: int, embedding_model_id: str = DEFAULT_EMBEDDING_MODEL
) -> dict[str, Any]:
    """The ``extra_params["rag"]`` block for a live RAG experiment over the SQuAD subset."""
    if top_k <= 0:
        raise ValueError("Retrieval top_k must be a positive integer.")
    subset = load_squad_subset()
    return RagPipelineConfig(
        embedding_model_id=embedding_model_id,
        chunk_size=SQUAD_CHUNKING.chunk_size,
        chunk_overlap=SQUAD_CHUNKING.chunk_overlap,
        top_k=top_k,
        corpus_fingerprint=subset.corpus_fingerprint(),
        prompt_template_id=RAG_PROMPT_TEMPLATE,
    ).to_dict()


# --- Adapters ----------------------------------------------------------------


def build_live_rag_pipeline(
    model_config: ModelConfig,
    allow_external_inference: bool,
    generation_client: Any | None = None,
    embedding_client: Any | None = None,
) -> ModelAdapter:
    """Rebuild the RAG pipeline an experiment declares, with real models."""
    _require_opt_in(allow_external_inference, "The live RAG pipeline")
    declared = model_config.extra_params.get("rag")
    if not isinstance(declared, dict):
        raise ValueError(
            "This experiment declares no retrieval configuration (model_config.extra_params.rag). "
            "Use the closed-book adapter 'openai-custom', or create the experiment with retrieval."
        )
    config = RagPipelineConfig.from_dict(declared)
    corpus = known_corpora().get(config.corpus_fingerprint)
    if corpus is None:
        raise ValueError(
            f"Unknown corpus fingerprint {config.corpus_fingerprint[:12]}...: the live RAG "
            "pipeline can only rebuild retrieval over bundled corpora (the SQuAD v1.1 subset)."
        )
    chunks = corpus.chunks(ChunkingConfig(config.chunk_size, config.chunk_overlap))
    retriever = Retriever(
        chunks=chunks,
        embedding_model=cached_embedder(
            config.embedding_model_id, allow_external_inference, embedding_client
        ),
        top_k=config.top_k,
        similarity_metric=config.similarity_metric,
    )
    return RagPipeline(
        retriever=retriever,
        generation_adapter=openai_compatible_provider(
            model_config.model_id, allow_external_inference, generation_client
        ),
        config=config,
    )


def build_live_closed_book_adapter(
    model_config: ModelConfig, allow_external_inference: bool, client: Any | None = None
) -> ModelAdapter:
    """Plain OpenAI-compatible generation, wrapped in the declared prompt template if any."""
    provider = openai_compatible_provider(model_config.model_id, allow_external_inference, client)
    template_id = model_config.extra_params.get("prompt_template_id")
    if template_id:
        return PromptTemplateAdapter(provider, str(template_id))
    return provider


# --- Judges ------------------------------------------------------------------


class _DefinitionOnlyClient:
    """Stands in for an OpenAI client on judges built only to record their config."""

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError("This judge was built only to define an experiment; it cannot run.")


def judge_model_config(judge_model_id: str) -> ModelConfig:
    return ModelConfig(
        model_id=judge_model_id,
        temperature=JUDGE_TEMPERATURE,
        max_tokens=JUDGE_MAX_TOKENS,
        extra_params={"seed": JUDGE_SEED, "response_format": {"type": "json_object"}},
    )


def is_live_judge_config(parameters: dict[str, Any]) -> bool:
    """True when a declared judge configuration names a real (non-``fake-``) model."""
    model_id = parameters.get("judge_model_id")
    return isinstance(model_id, str) and not model_id.startswith("fake-")


def build_live_judge(
    name: str,
    allow_external_inference: bool,
    judge_model_id: str | None = None,
    declared: dict[str, Any] | None = None,
    client: Any | None = None,
    for_execution: bool = True,
    rubric_version: str = "v1",
) -> Any:
    """Build a real LLM judge, either for a new experiment or from a declared config.

    ``rubric_version`` applies to new experiments; a declared configuration is
    rebuilt with whichever known rubric it recorded.

    ``for_execution=False`` builds it only to record its configuration in a new
    experiment definition; the provider still refuses requests without opt-in.
    """
    versions = JUDGE_RUBRIC_VERSIONS.get(name)
    if versions is None:
        raise ValueError(f"'{name}' has no live judge.")
    if rubric_version not in versions:
        raise ValueError(f"Unknown rubric version '{rubric_version}' for '{name}'.")
    rubric = versions[rubric_version]
    if declared is not None:
        declared_rubric = declared.get("rubric")
        matching = [r for r in versions.values() if r.to_dict() == declared_rubric]
        if not matching:
            known = ", ".join(r.rubric_id for r in versions.values())
            raise ValueError(
                f"The experiment's '{name}' rubric is not one this platform version can "
                f"reproduce exactly (known: {known}), so the judge cannot be reproduced. "
                "Define a new experiment."
            )
        rubric = matching[0]
        config = ModelConfig.from_dict(declared["judge_model_config"])
    elif judge_model_id:
        config = judge_model_config(judge_model_id)
    else:
        raise ValueError(f"'{name}' needs a judge model id.")
    if for_execution:
        _require_opt_in(allow_external_inference, f"The '{name}' judge ({config.model_id})")
    if not for_execution and client is None:
        # Only the configuration is needed; no server has to be configured.
        client = _DefinitionOnlyClient()
    adapter = openai_compatible_provider(config.model_id, allow_external_inference, client)
    if name == "llm_judge":
        return LLMJudgeEvaluator(judge_adapter=adapter, judge_model_config=config, rubric=rubric)
    return LLMFaithfulnessEvaluator(judge_adapter=adapter, judge_model_config=config, rubric=rubric)


# --- Warm-up -----------------------------------------------------------------

LOCAL_ADAPTERS = frozenset({"openai-custom", "openai-custom-rag"})


def warm_up(
    generator_model_ids: Sequence[str],
    embedding_model_ids: Sequence[str],
    allow_external_inference: bool,
) -> None:
    """Send one unrecorded request per model before a run starts.

    Local servers load a model on first use (Ollama also unloads it after
    five idle minutes), so without this the first recorded latency of a run
    includes seconds of model loading (3.4 s vs ~0.3 s observed). Failures are
    ignored here: the run itself will surface them per test case.
    """
    if not allow_external_inference:
        return
    for model_id in dict.fromkeys(generator_model_ids):
        try:
            openai_compatible_provider(model_id, allow_external_inference).generate(
                ExecutionRequest("warm-up", "Say OK.", ModelConfig(model_id, 0.0, 4))
            )
        except Exception:  # noqa: S112 - best effort by design
            continue
    for embedding_model_id in dict.fromkeys(embedding_model_ids):
        try:
            cached_embedder(embedding_model_id, allow_external_inference).embed_one("warm-up")
        except Exception:  # noqa: S112 - best effort by design
            continue
