"""
Retrieval: ranking a chunk corpus against a query by embedding similarity.

``Retriever`` embeds its chunk corpus once at construction time, then for
each query embeds the query and ranks every chunk by cosine similarity,
returning the top-k as a ``RetrievalResult`` with stable chunk/document
identifiers and similarity scores, in ranked order. No vector database or
approximate index is used: every candidate is scored exactly, which is
appropriate for the small, controlled corpora this milestone targets. If
a future experiment demonstrates an actual performance requirement beyond
this, a vector index can be introduced later; it is not needed now.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from llm_reliability.rag.documents import Chunk
from llm_reliability.rag.embeddings import EmbeddingModel
from llm_reliability.rag.similarity import cosine_similarity


@dataclass
class RetrievalConfig:
    """Serializable configuration identifying how retrieval was performed."""

    embedding_model_id: str
    top_k: int
    similarity_metric: str = "cosine"

    def __post_init__(self) -> None:
        if not self.embedding_model_id or not self.embedding_model_id.strip():
            raise ValueError("RetrievalConfig.embedding_model_id must be a non-empty string.")
        if self.top_k <= 0:
            raise ValueError("RetrievalConfig.top_k must be a positive integer.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "embedding_model_id": self.embedding_model_id,
            "top_k": self.top_k,
            "similarity_metric": self.similarity_metric,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrievalConfig:
        return cls(
            embedding_model_id=data["embedding_model_id"],
            top_k=data["top_k"],
            similarity_metric=data.get("similarity_metric", "cosine"),
        )


@dataclass
class RetrievedChunk:
    """One ranked retrieval hit: a chunk plus its rank and similarity score."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    rank: int

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("RetrievedChunk.rank must be a positive integer (1-based).")

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "score": self.score,
            "rank": self.rank,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrievedChunk:
        return cls(
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            text=data["text"],
            score=data["score"],
            rank=data["rank"],
        )


@dataclass
class RetrievalResult:
    """The complete, ranked outcome of retrieving for one query."""

    query: str
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    config: RetrievalConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "retrieved_chunks": [rc.to_dict() for rc in self.retrieved_chunks],
            "config": self.config.to_dict() if self.config is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrievalResult:
        config_data = data.get("config")
        return cls(
            query=data["query"],
            retrieved_chunks=[
                RetrievedChunk.from_dict(rc) for rc in data.get("retrieved_chunks", [])
            ],
            config=RetrievalConfig.from_dict(config_data) if config_data is not None else None,
        )


class Retriever:
    """Embeds a fixed chunk corpus and ranks it against queries by cosine similarity."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedding_model: EmbeddingModel,
        top_k: int,
        similarity_metric: str = "cosine",
    ) -> None:
        if len(chunks) == 0:
            raise ValueError("Retriever requires at least one chunk in its corpus.")
        if similarity_metric != "cosine":
            raise ValueError(
                f"Retriever only supports similarity_metric='cosine' in this milestone, "
                f"got {similarity_metric!r}."
            )
        self._chunks = list(chunks)
        self._embedding_model = embedding_model
        self._config = RetrievalConfig(
            embedding_model_id=embedding_model.model_id,
            top_k=top_k,
            similarity_metric=similarity_metric,
        )
        self._chunk_vectors = embedding_model.embed([chunk.text for chunk in self._chunks])

    @property
    def config(self) -> RetrievalConfig:
        return self._config

    def retrieve(self, query: str) -> RetrievalResult:
        if not query or not query.strip():
            raise ValueError("Retriever.retrieve requires a non-empty query.")

        query_vector = self._embedding_model.embed_one(query)
        scored = [
            (cosine_similarity(query_vector, vector), chunk)
            for vector, chunk in zip(self._chunk_vectors, self._chunks, strict=True)
        ]
        # Stable sort: ties keep the original corpus order, so results are
        # fully deterministic for deterministic embeddings.
        scored.sort(key=lambda pair: pair[0], reverse=True)

        top = scored[: self._config.top_k]
        retrieved_chunks = [
            RetrievedChunk(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                text=chunk.text,
                score=score,
                rank=rank,
            )
            for rank, (score, chunk) in enumerate(top, start=1)
        ]
        return RetrievalResult(query=query, retrieved_chunks=retrieved_chunks, config=self._config)
