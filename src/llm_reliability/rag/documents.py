"""
Documents and chunks: the retrieval corpus's domain models.

A ``Document`` is a single source text with a stable identifier. A
``Chunk`` is a piece of a document produced by a chunking strategy, and
always retains its source document id and position so retrieved evidence
can be traced back to exactly where it came from. ``FixedSizeChunker`` is
the only chunking strategy in this milestone: a deterministic, fixed
character-window splitter. Semantic, hierarchical, or recursive chunking
are explicitly out of scope; the objective here is a small, reproducible
baseline, not a general-purpose ingestion system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A single source document with a stable identifier."""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise ValueError("Document.id must be a non-empty string.")
        if not self.content or not self.content.strip():
            raise ValueError(f"Document '{self.id}': content must be a non-empty string.")

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "content": self.content, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        return cls(id=data["id"], content=data["content"], metadata=dict(data.get("metadata", {})))


@dataclass
class Chunk:
    """A piece of a document, produced by a chunking strategy.

    ``position`` is the zero-based index of this chunk within its source
    document's chunk sequence, so the original order can always be
    recovered even after chunks from many documents are pooled into one
    retrieval corpus.
    """

    id: str
    document_id: str
    text: str
    position: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise ValueError("Chunk.id must be a non-empty string.")
        if not self.document_id or not self.document_id.strip():
            raise ValueError(f"Chunk '{self.id}': document_id must be a non-empty string.")
        if not self.text or not self.text.strip():
            raise ValueError(f"Chunk '{self.id}': text must be a non-empty string.")
        if self.position < 0:
            raise ValueError(f"Chunk '{self.id}': position must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "text": self.text,
            "position": self.position,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        return cls(
            id=data["id"],
            document_id=data["document_id"],
            text=data["text"],
            position=data["position"],
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ChunkingConfig:
    """Configuration for ``FixedSizeChunker``.

    ``chunk_size`` is the maximum number of characters per chunk (the
    explicit unit is characters, not tokens or words, to keep behavior
    simple and dependency-free). ``chunk_overlap`` is the number of
    characters repeated at the start of each chunk from the end of the
    previous one, and must be strictly smaller than ``chunk_size`` so that
    chunking always makes forward progress.
    """

    chunk_size: int
    chunk_overlap: int = 0

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("ChunkingConfig.chunk_size must be a positive integer.")
        if self.chunk_overlap < 0:
            raise ValueError("ChunkingConfig.chunk_overlap must be non-negative.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "ChunkingConfig.chunk_overlap must be strictly smaller than chunk_size."
            )

    def to_dict(self) -> dict[str, Any]:
        return {"chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkingConfig:
        return cls(chunk_size=data["chunk_size"], chunk_overlap=data.get("chunk_overlap", 0))


class FixedSizeChunker:
    """Deterministic, fixed character-window chunker.

    Splits a document's content into consecutive windows of at most
    ``chunk_size`` characters, advancing by ``chunk_size - chunk_overlap``
    characters each step. The same document and configuration always
    produce the same chunks: there is no randomness and no dependence on
    anything other than the input text and configuration.
    """

    def __init__(self, config: ChunkingConfig) -> None:
        self._config = config

    def get_config(self) -> dict[str, Any]:
        return self._config.to_dict()

    def chunk_document(self, document: Document) -> list[Chunk]:
        text = document.content
        size = self._config.chunk_size
        step = size - self._config.chunk_overlap

        chunks: list[Chunk] = []
        start = 0
        position = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + size, text_length)
            chunks.append(
                Chunk(
                    id=f"{document.id}::chunk-{position}",
                    document_id=document.id,
                    text=text[start:end],
                    position=position,
                )
            )
            if end == text_length:
                break
            start += step
            position += 1
        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self.chunk_document(document))
        return chunks
