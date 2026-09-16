"""Tests for the Retriever: ranking, top-k, determinism, and configuration."""

import pytest

from llm_reliability.rag.documents import Chunk
from llm_reliability.rag.embeddings import FakeEmbeddingModel
from llm_reliability.rag.retriever import RetrievalConfig, Retriever


def _chunk(chunk_id: str, document_id: str, text: str) -> Chunk:
    return Chunk(id=chunk_id, document_id=document_id, text=text, position=0)


class TestRetrievalConfig:
    def test_requires_non_empty_embedding_model_id(self):
        with pytest.raises(ValueError, match="embedding_model_id"):
            RetrievalConfig(embedding_model_id="", top_k=3)

    def test_requires_positive_top_k(self):
        with pytest.raises(ValueError, match="top_k"):
            RetrievalConfig(embedding_model_id="m1", top_k=0)

    def test_round_trip_through_dict(self):
        config = RetrievalConfig(embedding_model_id="m1", top_k=5)
        assert RetrievalConfig.from_dict(config.to_dict()) == config


class TestRetriever:
    def test_requires_at_least_one_chunk(self):
        with pytest.raises(ValueError, match="at least one chunk"):
            Retriever(chunks=[], embedding_model=FakeEmbeddingModel(), top_k=3)

    def test_only_cosine_similarity_is_supported(self):
        chunks = [_chunk("c1", "d1", "text")]
        with pytest.raises(ValueError, match="cosine"):
            Retriever(
                chunks=chunks,
                embedding_model=FakeEmbeddingModel(),
                top_k=1,
                similarity_metric="euclidean",
            )

    def test_ranks_most_similar_chunk_first(self):
        chunks = [
            _chunk("c-relevant", "d1", "Paris is the capital of France."),
            _chunk("c-irrelevant", "d2", "Bananas are a good source of potassium."),
        ]
        embedding_model = FakeEmbeddingModel(
            vectors={
                "What is the capital of France?": [1, 0, 0, 0, 0, 0, 0, 0],
                "Paris is the capital of France.": [1, 0, 0, 0, 0, 0, 0, 0],
                "Bananas are a good source of potassium.": [0, 1, 0, 0, 0, 0, 0, 0],
            }
        )
        retriever = Retriever(chunks=chunks, embedding_model=embedding_model, top_k=2)
        result = retriever.retrieve("What is the capital of France?")

        assert result.retrieved_chunks[0].chunk_id == "c-relevant"
        assert result.retrieved_chunks[0].rank == 1
        assert result.retrieved_chunks[1].chunk_id == "c-irrelevant"
        assert result.retrieved_chunks[1].rank == 2
        assert result.retrieved_chunks[0].score > result.retrieved_chunks[1].score

    def test_top_k_truncates_without_reordering(self):
        chunks = [_chunk(f"c{i}", "d1", f"chunk text {i}") for i in range(5)]
        retriever = Retriever(chunks=chunks, embedding_model=FakeEmbeddingModel(), top_k=2)
        result = retriever.retrieve("a query")
        assert len(result.retrieved_chunks) == 2
        assert result.retrieved_chunks[0].rank == 1
        assert result.retrieved_chunks[1].rank == 2

    def test_top_k_larger_than_corpus_returns_all_chunks(self):
        chunks = [_chunk("c1", "d1", "only chunk")]
        retriever = Retriever(chunks=chunks, embedding_model=FakeEmbeddingModel(), top_k=10)
        result = retriever.retrieve("a query")
        assert len(result.retrieved_chunks) == 1

    def test_retrieval_is_deterministic(self):
        chunks = [_chunk(f"c{i}", "d1", f"chunk text {i}") for i in range(5)]
        embedding_model = FakeEmbeddingModel()
        retriever = Retriever(chunks=chunks, embedding_model=embedding_model, top_k=3)
        first = retriever.retrieve("a query")
        second = retriever.retrieve("a query")
        assert first.to_dict() == second.to_dict()

    def test_result_records_retrieval_configuration(self):
        chunks = [_chunk("c1", "d1", "text")]
        embedding_model = FakeEmbeddingModel()
        retriever = Retriever(chunks=chunks, embedding_model=embedding_model, top_k=3)
        result = retriever.retrieve("a query")
        assert result.config.embedding_model_id == embedding_model.model_id
        assert result.config.top_k == 3
        assert result.config.similarity_metric == "cosine"

    def test_stable_chunk_and_document_identifiers_are_preserved(self):
        chunks = [_chunk("c1", "doc-1", "text")]
        retriever = Retriever(chunks=chunks, embedding_model=FakeEmbeddingModel(), top_k=1)
        result = retriever.retrieve("a query")
        assert result.retrieved_chunks[0].chunk_id == "c1"
        assert result.retrieved_chunks[0].document_id == "doc-1"

    def test_empty_query_is_rejected(self):
        chunks = [_chunk("c1", "d1", "text")]
        retriever = Retriever(chunks=chunks, embedding_model=FakeEmbeddingModel(), top_k=1)
        with pytest.raises(ValueError, match="non-empty query"):
            retriever.retrieve("")

    def test_does_not_fabricate_scores_beyond_the_similarity_function(self):
        chunks = [
            _chunk("c1", "d1", "same"),
            _chunk("c2", "d1", "same"),
        ]
        embedding_model = FakeEmbeddingModel(
            vectors={"same": [1, 0, 0, 0, 0, 0, 0, 0], "same query": [1, 0, 0, 0, 0, 0, 0, 0]}
        )
        retriever = Retriever(chunks=chunks, embedding_model=embedding_model, top_k=2)
        result = retriever.retrieve("same query")
        # Identical embeddings must score identically -- no arbitrary tie-break score shift.
        assert result.retrieved_chunks[0].score == pytest.approx(result.retrieved_chunks[1].score)

    def test_round_trip_result_through_dict(self):
        chunks = [_chunk("c1", "d1", "text")]
        retriever = Retriever(chunks=chunks, embedding_model=FakeEmbeddingModel(), top_k=1)
        result = retriever.retrieve("a query")
        from llm_reliability.rag.retriever import RetrievalResult

        reloaded = RetrievalResult.from_dict(result.to_dict())
        assert reloaded == result
