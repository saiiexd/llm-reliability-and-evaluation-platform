"""
Allowlisted adapter and evaluator registries for safe, controlled execution.

Prevents arbitrary Python code execution, dynamic imports, shell commands,
or unvalidated object construction from client requests.
Supports both deterministic local mock/RAG adapters and allowlisted real LLM providers.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from llm_reliability.evaluation.adapters import MockAdapter, ModelAdapter
from llm_reliability.evaluation.evaluators import ExactMatchEvaluator
from llm_reliability.evaluation.models import ModelConfig
from llm_reliability.evaluation.semantic_similarity import (
    FakeSimilarityBackend,
    SemanticSimilarityEvaluator,
)
from llm_reliability.providers.base import ProviderConfig
from llm_reliability.providers.openai_provider import (
    OpenAIProviderAdapter,
    get_openai_api_key_from_env,
    get_openai_base_url_from_env,
)
from llm_reliability.rag.documents import ChunkingConfig, Document, FixedSizeChunker
from llm_reliability.rag.embeddings import FakeEmbeddingModel
from llm_reliability.rag.faithfulness_judge import LLMFaithfulnessEvaluator
from llm_reliability.rag.pipeline import (
    RagPipeline,
    RagPipelineConfig,
    compute_corpus_fingerprint,
)
from llm_reliability.rag.retriever import Retriever

GENERATION_RESPONSES = {
    "correct-context-correct-answer": "Paris",
    "correct-context-unsupported-answer": "Paris, which has a population of 30 million.",
    "correct-context-contradictory-answer": "Lyon is the capital of France.",
    "relevant-context-incorrect-answer": "Berlin",
    "missing-context": "I do not know.",
    "no-ground-truth": "Paris",
    "correct-context-correct-answer-reworded": "Paris",
}

JUDGE_RESPONSES = {
    # Canned judge replies for the deterministic fake judge. They must follow the
    # exact output contract of FAITHFULNESS_RUBRIC ({"outcome", "reasoning"}),
    # because parse_judge_output is deliberately strict. Any test case id not
    # listed here makes MockAdapter echo the prompt, which the parser rejects as
    # an output_validation_error -- the fake judge only "knows" these cases.
    "correct-context-correct-answer": (
        '{"outcome": "supported", "reasoning": "Directly supported by the context."}'
    ),
    "correct-context-unsupported-answer": (
        '{"outcome": "unsupported", "reasoning": "The context does not mention population."}'
    ),
    "correct-context-contradictory-answer": (
        '{"outcome": "contradicted", "reasoning": "The context says Paris; the answer says Lyon."}'
    ),
    "relevant-context-incorrect-answer": (
        '{"outcome": "unsupported", "reasoning": "Berlin is not Paris."}'
    ),
    "missing-context": (
        '{"outcome": "unsupported", "reasoning": "No relevant context was retrieved."}'
    ),
    "no-ground-truth": ('{"outcome": "supported", "reasoning": "Supported by the context."}'),
    "correct-context-correct-answer-reworded": (
        '{"outcome": "supported", "reasoning": "Directly supported by the context."}'
    ),
}


def _build_mock_rag_pipeline(top_k: int = 1) -> ModelAdapter:
    docs = [
        Document(id="doc-france", content="Paris is the capital of France."),
        Document(id="doc-fruit", content="Bananas are a good source of potassium."),
    ]
    chunks = FixedSizeChunker(ChunkingConfig(chunk_size=200, chunk_overlap=0)).chunk_documents(docs)
    embeddings = FakeEmbeddingModel(
        vectors={
            "What is the capital of France?": [1, 0],
            "Which city serves as the capital of France?": [1, 0],
            "What is the capital of a country not in this corpus?": [0, 1],
            "Paris is the capital of France.": [1, 0],
            "Bananas are a good source of potassium.": [0, 1],
        },
        dimensions=2,
    )
    retriever = Retriever(chunks=chunks, embedding_model=embeddings, top_k=top_k)
    return RagPipeline(
        retriever=retriever,
        generation_adapter=MockAdapter(
            responses=GENERATION_RESPONSES,
            model_id="fake-rag-generator",
        ),
        config=RagPipelineConfig(
            embedding_model_id="fake-embedding-v1",
            chunk_size=200,
            chunk_overlap=0,
            top_k=top_k,
            corpus_fingerprint=compute_corpus_fingerprint(docs),
        ),
    )


class AdapterRegistry:
    """Registry of allowlisted, controlled model and pipeline adapters."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[dict[str, Any] | None], ModelAdapter]] = {
            "mock-adapter-v1": lambda cfg: MockAdapter(
                responses=GENERATION_RESPONSES,
                model_id="mock-adapter-v1",
            ),
            "rag-pipeline-v1": lambda cfg: _build_mock_rag_pipeline(
                top_k=cfg.get("top_k", 1) if cfg else 1
            ),
            "rag-pipeline-topk-2": lambda cfg: _build_mock_rag_pipeline(top_k=2),
            "echo-adapter": lambda cfg: MockAdapter(model_id="echo-adapter"),
            "openai-gpt-4o-mini": lambda cfg: OpenAIProviderAdapter(
                config=ProviderConfig(
                    provider_id="openai",
                    model_id="gpt-4o-mini",
                    base_url=(cfg.get("base_url") if cfg else None),
                    timeout_seconds=float(cfg.get("timeout_seconds", 30.0)) if cfg else 30.0,
                    max_retries=int(cfg.get("max_retries", 3)) if cfg else 3,
                    allow_external_inference=bool(cfg.get("allow_external_inference", False))
                    if cfg
                    else False,
                ),
                client=cfg.get("client") if cfg else None,
                sleeper=cfg.get("sleeper") if cfg else None,
            ),
            "openai-gpt-4o": lambda cfg: OpenAIProviderAdapter(
                config=ProviderConfig(
                    provider_id="openai",
                    model_id="gpt-4o",
                    base_url=(cfg.get("base_url") if cfg else None),
                    timeout_seconds=float(cfg.get("timeout_seconds", 30.0)) if cfg else 30.0,
                    max_retries=int(cfg.get("max_retries", 3)) if cfg else 3,
                    allow_external_inference=bool(cfg.get("allow_external_inference", False))
                    if cfg
                    else False,
                ),
                client=cfg.get("client") if cfg else None,
                sleeper=cfg.get("sleeper") if cfg else None,
            ),
            # Generic OpenAI-compatible endpoint (Ollama, vLLM, LM Studio). The
            # served model name comes from the experiment's model_config.model_id
            # and the URL from LLM_RELIABILITY_OPENAI_BASE_URL / OPENAI_BASE_URL.
            "openai-custom": lambda cfg: OpenAIProviderAdapter(
                use_request_model=True,
                config=ProviderConfig(
                    provider_id="openai-compatible",
                    model_id=str(cfg.get("model_id", "custom-model")) if cfg else "custom-model",
                    base_url=(cfg.get("base_url") if cfg else None),
                    timeout_seconds=float(cfg.get("timeout_seconds", 30.0)) if cfg else 30.0,
                    max_retries=int(cfg.get("max_retries", 3)) if cfg else 3,
                    allow_external_inference=bool(cfg.get("allow_external_inference", False))
                    if cfg
                    else False,
                ),
                client=cfg.get("client") if cfg else None,
                sleeper=cfg.get("sleeper") if cfg else None,
            ),
        }

    def list_available(self) -> list[str]:
        return sorted(self._factories.keys())

    def get(self, adapter_id: str, config: dict[str, Any] | None = None) -> ModelAdapter:
        if adapter_id not in self._factories:
            available = ", ".join(self.list_available())
            raise ValueError(f"Unknown adapter '{adapter_id}'. Allowlisted adapters: {available}")
        return self._factories[adapter_id](config)

    def is_external_provider(self, adapter_id: str) -> bool:
        """Indicate whether this adapter connects to an external live provider."""
        return adapter_id.startswith("openai-")


class EvaluatorRegistry:
    """Registry of allowlisted evaluator instances."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], Any]] = {
            "exact_match": lambda: ExactMatchEvaluator(),
            "semantic_similarity": lambda: SemanticSimilarityEvaluator(
                backend=FakeSimilarityBackend()
            ),
            "llm_faithfulness_judge": lambda: LLMFaithfulnessEvaluator(
                judge_adapter=MockAdapter(
                    responses=JUDGE_RESPONSES,
                    model_id="fake-judge-v1",
                ),
                judge_model_config=ModelConfig(model_id="fake-judge-v1"),
            ),
        }

    # Retrieval metrics are not answer evaluators: they return a
    # RetrievalEvaluationResult (no ``passed``), score the retriever rather than
    # the answer, and are computed automatically for every RAG run by
    # DiagnosticsService. Accepting them here used to persist a run whose
    # aggregation then crashed with an AttributeError (HTTP 500).
    RETRIEVAL_METRIC_NAMES = frozenset({"hit_at_1", "recall_at_1", "mrr"})

    def list_available(self) -> list[str]:
        return sorted(self._factories.keys())

    def get(self, name: str) -> Any:
        if name in self.RETRIEVAL_METRIC_NAMES:
            raise ValueError(
                f"'{name}' is a retrieval metric, not an answer evaluator. Retrieval metrics "
                "(Hit@K, Recall@K, MRR) are computed automatically for every RAG run and "
                "shown under retrieval results; remove it from evaluator_names."
            )
        if name not in self._factories:
            available = ", ".join(self.list_available())
            raise ValueError(f"Unknown evaluator '{name}'. Allowlisted evaluators: {available}")
        return self._factories[name]()


def get_providers_metadata() -> list[dict[str, Any]]:
    """Return provider metadata and configuration readiness without disclosing secrets."""
    openai_key_present = bool(get_openai_api_key_from_env())
    openai_base_url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"

    return [
        {
            "provider_id": "openai",
            "name": "OpenAI",
            "is_configured": openai_key_present,
            "base_url": openai_base_url,
            "requires_api_key": True,
            "supported_models": [
                {
                    "model_id": "gpt-4o-mini",
                    "display_name": "GPT-4o Mini",
                    "context_window": 128000,
                    "default_temperature": 0.0,
                },
                {
                    "model_id": "gpt-4o",
                    "display_name": "GPT-4o",
                    "context_window": 128000,
                    "default_temperature": 0.0,
                },
            ],
            "default_model": "gpt-4o-mini",
        },
        {
            "provider_id": "openai-compatible",
            "name": "OpenAI-Compatible (Local / Self-Hosted)",
            "is_configured": True,
            "base_url": get_openai_base_url_from_env(),
            "requires_api_key": False,
            "supported_models": [
                {
                    "model_id": "custom-model",
                    "display_name": "Custom Model (Ollama / vLLM / LM Studio)",
                    "context_window": 32768,
                    "default_temperature": 0.0,
                }
            ],
            "default_model": "custom-model",
        },
    ]
