"""
RAG pipeline: composes retrieval with generation.

Implements the explicit execution model this milestone is built around::

    Question -> Retrieval -> Retrieved Context -> Generation -> Answer

``RagPipeline`` implements the existing ``ModelAdapter`` interface (from
``llm_reliability.evaluation``) rather than introducing a parallel
execution abstraction: retrieval and generation are functionally just
another way to produce a ``ModelResponse`` from an ``ExecutionRequest``.
This means ``EvaluationRunner`` and ``ExperimentManager`` need no changes
at all to run a RAG experiment -- an experiment simply supplies a
``RagPipeline`` wherever it would otherwise supply a plain adapter.

Retrieval evidence (the retrieved chunks, their ranks and scores, and the
retrieval configuration) is carried in the returned
``ModelResponse.provider_metadata`` under the ``"rag"`` key. This is
exactly the extension point ``ModelResponse`` was designed with in Prompt
1 ("a future RAG adapter can carry retrieved-context information there
without forcing retrieval fields into every adapter's schema"), so no
change to the Core Evaluation Engine's domain models was needed either.

Prompt construction is explicit and inspectable: ``build_rag_prompt``
renders the exact text sent to the generation adapter, and that rendered
text is also retained in ``provider_metadata["rag"]["prompt"]`` so it can
be audited after the fact, not just trusted to have been built correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_reliability.evaluation.adapters import ModelAdapter
from llm_reliability.evaluation.models import ExecutionRequest, ModelResponse
from llm_reliability.experiments.fingerprint import content_fingerprint
from llm_reliability.rag.documents import Document
from llm_reliability.rag.retriever import RetrievedChunk, Retriever

PROMPT_TEMPLATE_ID = "default_v1"


def compute_corpus_fingerprint(documents: list[Document]) -> str:
    """Deterministic content fingerprint of a document corpus.

    Reuses the same canonical-JSON/SHA-256 strategy as dataset and
    experiment fingerprinting (``llm_reliability.experiments.fingerprint``)
    so that changing the underlying document corpus is detectable the same
    way changing a dataset is. Computed from document id and content only
    (not metadata, and not chunking), ordered by document id so the
    fingerprint does not depend on corpus list order.
    """
    payload = [
        {"id": doc.id, "content": doc.content} for doc in sorted(documents, key=lambda d: d.id)
    ]
    return content_fingerprint({"documents": payload})


@dataclass
class RagPipelineConfig:
    """Serializable configuration for one RAG pipeline: everything that affects retrieval.

    Callers defining a RAG experiment should embed
    ``RagPipelineConfig.to_dict()`` in the experiment's
    ``ModelConfig.extra_params`` (for example, under an ``"rag"`` key) so it
    participates in the experiment configuration fingerprint (see
    ``llm_reliability.experiments``) exactly like any other provider-specific
    setting -- no change to the experiment system itself is required for
    this to work.
    """

    embedding_model_id: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    corpus_fingerprint: str
    similarity_metric: str = "cosine"
    prompt_template_id: str = PROMPT_TEMPLATE_ID

    def to_dict(self) -> dict[str, Any]:
        return {
            "embedding_model_id": self.embedding_model_id,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "corpus_fingerprint": self.corpus_fingerprint,
            "similarity_metric": self.similarity_metric,
            "prompt_template_id": self.prompt_template_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RagPipelineConfig:
        return cls(
            embedding_model_id=data["embedding_model_id"],
            chunk_size=data["chunk_size"],
            chunk_overlap=data["chunk_overlap"],
            top_k=data["top_k"],
            corpus_fingerprint=data["corpus_fingerprint"],
            similarity_metric=data.get("similarity_metric", "cosine"),
            prompt_template_id=data.get("prompt_template_id", PROMPT_TEMPLATE_ID),
        )


def build_rag_prompt(question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    """Render the exact generation prompt from a question and ranked retrieved chunks.

    This is ``prompt_template_id="default_v1"``: numbered context passages
    in ranked order (each labeled with its source document id), followed
    by the question and an explicit instruction not to answer beyond the
    given context. Context construction happens here, in the open, rather
    than inside the retriever or the model adapter.
    """
    if not retrieved_chunks:
        context_block = "(no context retrieved)"
    else:
        context_block = "\n".join(
            f"[{chunk.rank}] (source: {chunk.document_id}) {chunk.text}"
            for chunk in retrieved_chunks
        )
    return (
        "Answer the question using only the context passages below. "
        "If the context does not contain the answer, say you do not know.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
    )


class RagPipeline(ModelAdapter):
    """Composes a ``Retriever`` with a generation ``ModelAdapter``.

    Retrieval and generation remain distinct stages: this class calls the
    retriever exactly once per request, builds the prompt explicitly, then
    calls the generation adapter exactly once with that prompt. If either
    step raises, the exception propagates unchanged; ``EvaluationRunner``
    already isolates such failures to the affected test case, so no
    additional error handling is introduced here.
    """

    def __init__(
        self, retriever: Retriever, generation_adapter: ModelAdapter, config: RagPipelineConfig
    ) -> None:
        self._retriever = retriever
        self._generation_adapter = generation_adapter
        self._config = config

    @property
    def config(self) -> RagPipelineConfig:
        return self._config

    def generate(self, request: ExecutionRequest) -> ModelResponse:
        retrieval_result = self._retriever.retrieve(request.input_text)
        prompt = build_rag_prompt(request.input_text, retrieval_result.retrieved_chunks)

        generation_request = ExecutionRequest(
            test_case_id=request.test_case_id,
            input_text=prompt,
            model_config=request.model_config,
        )
        response = self._generation_adapter.generate(generation_request)

        provider_metadata = dict(response.provider_metadata)
        provider_metadata["rag"] = {
            "query": retrieval_result.query,
            "retrieved_chunks": [chunk.to_dict() for chunk in retrieval_result.retrieved_chunks],
            "retrieval_config": retrieval_result.config.to_dict()
            if retrieval_result.config
            else None,
            "pipeline_config": self._config.to_dict(),
            "prompt": prompt,
        }
        return ModelResponse(
            test_case_id=response.test_case_id,
            output_text=response.output_text,
            model_id=response.model_id,
            latency_ms=response.latency_ms,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            provider_metadata=provider_metadata,
        )


def extract_rag_metadata(response: ModelResponse) -> dict[str, Any] | None:
    """Return the ``"rag"`` metadata block from a response, or ``None`` if absent.

    ``None`` means this response was not produced by a ``RagPipeline`` (or
    the metadata was stripped), which downstream retrieval evaluators and
    diagnosis must treat as "no retrieval evidence available," never as a
    retrieval failure.
    """
    return response.provider_metadata.get("rag")


def extract_retrieved_chunks(response: ModelResponse) -> list[RetrievedChunk] | None:
    """Reconstruct the ranked ``RetrievedChunk`` list from a response's RAG metadata."""
    rag_metadata = extract_rag_metadata(response)
    if rag_metadata is None:
        return None
    return [RetrievedChunk.from_dict(chunk) for chunk in rag_metadata.get("retrieved_chunks", [])]
