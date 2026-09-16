"""
Embedding abstraction for retrieval.

``EmbeddingModel`` separates the retrieval engine from any specific
embedding model or provider: the retriever depends only on this
interface, never on a concrete library. ``FakeEmbeddingModel`` is
deterministic and requires no dependency, so the standard test suite can
exercise the full retrieval pipeline without network access or model
downloads. ``SentenceTransformerEmbeddingModel`` is the one real
implementation for this milestone (per the project's research plan);
Sentence Transformers was already available in this environment, so no
new dependency was required to enable it, but it imports the library
lazily -- only inside ``embed()``, never at module import or construction
time -- because the first real call downloads and caches a pretrained
model, which requires network access this project's standard tests must
never depend on.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any


class EmbeddingBackendError(Exception):
    """Raised when an embedding model cannot produce vectors in the current environment."""


class EmbeddingModel(ABC):
    """Converts text into numerical vectors for retrieval."""

    model_id: str

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in the same order."""
        raise NotImplementedError

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def get_config(self) -> dict[str, Any]:
        return {"model_id": self.model_id}


class FakeEmbeddingModel(EmbeddingModel):
    """Deterministic fake embedding model for tests and local development.

    Returns an explicitly configured vector for any text supplied in
    ``vectors`` (keyed by the exact string), which is how controlled
    retrieval scenarios are built: give the "relevant" chunk text and the
    query a similar configured vector, and an "irrelevant" chunk text a
    dissimilar one. Any text not present in ``vectors`` falls back to a
    deterministic, content-derived vector (from a SHA-256 digest of the
    text), so a corpus can be embedded without configuring every single
    chunk by hand while remaining fully deterministic. Performs no real
    embedding computation and must never be used to draw conclusions about
    real retrieval quality.
    """

    model_id = "fake-embedding-v1"

    def __init__(self, vectors: dict[str, list[float]] | None = None, dimensions: int = 8) -> None:
        if dimensions <= 0:
            raise ValueError("FakeEmbeddingModel.dimensions must be a positive integer.")
        self._vectors = dict(vectors) if vectors else {}
        self._dimensions = dimensions

    def get_config(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "dimensions": self._dimensions}

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        if text in self._vectors:
            configured = self._vectors[text]
            if len(configured) != self._dimensions:
                raise ValueError(
                    f"FakeEmbeddingModel: configured vector for {text!r} has "
                    f"{len(configured)} dimensions, expected {self._dimensions}."
                )
            return list(configured)
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i % len(digest)] / 255.0 for i in range(self._dimensions)]


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    """Real embedding backend using the ``sentence-transformers`` library.

    The underlying model is loaded lazily, on the first call to
    ``embed()``, never at construction time: constructing this class and
    reading its configuration never requires the library or a model
    download. This is why it is never exercised by the standard, offline
    test suite (see ``tests/rag/test_embeddings_integration.py``, which
    skips itself unless ``sentence-transformers`` is importable).
    """

    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_id = model_id
        self._model: Any = None

    def get_config(self) -> dict[str, Any]:
        return {"model_id": self.model_id}

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingBackendError(
                    "The 'sentence-transformers' package is required to compute real "
                    "embeddings but is not installed in this environment."
                ) from exc
            self._model = SentenceTransformer(self.model_id)
        vectors = self._model.encode(list(texts), convert_to_numpy=True)
        return [vector.tolist() for vector in vectors]
