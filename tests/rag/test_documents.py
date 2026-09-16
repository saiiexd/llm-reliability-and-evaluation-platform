"""Tests for Document, Chunk, ChunkingConfig, and FixedSizeChunker."""

import pytest

from llm_reliability.rag.documents import Chunk, ChunkingConfig, Document, FixedSizeChunker


class TestDocument:
    def test_requires_non_empty_id(self):
        with pytest.raises(ValueError, match="id"):
            Document(id="", content="text")

    def test_requires_non_empty_content(self):
        with pytest.raises(ValueError, match="content"):
            Document(id="d1", content="")

    def test_round_trip_through_dict(self):
        doc = Document(id="d1", content="hello world", metadata={"source": "test"})
        reloaded = Document.from_dict(doc.to_dict())
        assert reloaded == doc


class TestChunk:
    def test_requires_non_empty_text(self):
        with pytest.raises(ValueError, match="text"):
            Chunk(id="c1", document_id="d1", text="", position=0)

    def test_rejects_negative_position(self):
        with pytest.raises(ValueError, match="position"):
            Chunk(id="c1", document_id="d1", text="x", position=-1)

    def test_round_trip_through_dict(self):
        chunk = Chunk(id="c1", document_id="d1", text="hello", position=0)
        reloaded = Chunk.from_dict(chunk.to_dict())
        assert reloaded == chunk


class TestChunkingConfig:
    def test_rejects_non_positive_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size"):
            ChunkingConfig(chunk_size=0)

    def test_rejects_negative_overlap(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            ChunkingConfig(chunk_size=10, chunk_overlap=-1)

    def test_rejects_overlap_greater_than_or_equal_to_chunk_size(self):
        with pytest.raises(ValueError, match="overlap"):
            ChunkingConfig(chunk_size=10, chunk_overlap=10)

    def test_valid_config(self):
        config = ChunkingConfig(chunk_size=10, chunk_overlap=2)
        assert config.chunk_size == 10
        assert config.chunk_overlap == 2

    def test_round_trip_through_dict(self):
        config = ChunkingConfig(chunk_size=20, chunk_overlap=5)
        assert ChunkingConfig.from_dict(config.to_dict()) == config


class TestFixedSizeChunker:
    def test_short_document_produces_one_chunk(self):
        chunker = FixedSizeChunker(ChunkingConfig(chunk_size=100, chunk_overlap=0))
        doc = Document(id="d1", content="short text")
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "short text"
        assert chunks[0].id == "d1::chunk-0"
        assert chunks[0].position == 0

    def test_long_document_produces_multiple_chunks_without_overlap(self):
        chunker = FixedSizeChunker(ChunkingConfig(chunk_size=5, chunk_overlap=0))
        doc = Document(id="d1", content="0123456789")
        chunks = chunker.chunk_document(doc)
        assert [c.text for c in chunks] == ["01234", "56789"]
        assert [c.position for c in chunks] == [0, 1]

    def test_chunks_preserve_document_id(self):
        chunker = FixedSizeChunker(ChunkingConfig(chunk_size=5, chunk_overlap=0))
        doc = Document(id="my-doc", content="0123456789")
        chunks = chunker.chunk_document(doc)
        assert all(c.document_id == "my-doc" for c in chunks)

    def test_overlap_produces_repeated_characters(self):
        chunker = FixedSizeChunker(ChunkingConfig(chunk_size=5, chunk_overlap=2))
        doc = Document(id="d1", content="0123456789")
        chunks = chunker.chunk_document(doc)
        # step = 5 - 2 = 3: windows start at 0, 3, 6; the window starting at 6
        # reaches the end of the text, so chunking stops there.
        assert [c.text for c in chunks] == ["01234", "34567", "6789"]

    def test_chunking_is_deterministic(self):
        chunker = FixedSizeChunker(ChunkingConfig(chunk_size=7, chunk_overlap=3))
        doc = Document(id="d1", content="the quick brown fox jumps over the lazy dog")
        first = chunker.chunk_document(doc)
        second = chunker.chunk_document(doc)
        assert [c.to_dict() for c in first] == [c.to_dict() for c in second]

    def test_chunk_documents_pools_chunks_from_multiple_documents(self):
        chunker = FixedSizeChunker(ChunkingConfig(chunk_size=10, chunk_overlap=0))
        docs = [
            Document(id="d1", content="hello world"),
            Document(id="d2", content="goodbye world"),
        ]
        chunks = chunker.chunk_documents(docs)
        document_ids = {c.document_id for c in chunks}
        assert document_ids == {"d1", "d2"}

    def test_no_chunk_exceeds_configured_chunk_size(self):
        chunker = FixedSizeChunker(ChunkingConfig(chunk_size=6, chunk_overlap=1))
        doc = Document(id="d1", content="a" * 37)
        chunks = chunker.chunk_document(doc)
        assert all(len(c.text) <= 6 for c in chunks)

    def test_get_config_reflects_chunking_config(self):
        chunker = FixedSizeChunker(ChunkingConfig(chunk_size=8, chunk_overlap=2))
        assert chunker.get_config() == {"chunk_size": 8, "chunk_overlap": 2}
