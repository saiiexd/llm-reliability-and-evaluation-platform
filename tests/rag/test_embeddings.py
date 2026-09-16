"""Tests for the embedding abstraction and its deterministic fake."""

import pytest

from llm_reliability.rag.embeddings import FakeEmbeddingModel


class TestFakeEmbeddingModel:
    def test_configured_text_returns_configured_vector(self):
        model = FakeEmbeddingModel(vectors={"hello": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]})
        assert model.embed_one("hello") == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def test_unconfigured_text_is_deterministic(self):
        model = FakeEmbeddingModel()
        first = model.embed_one("some text never configured")
        second = model.embed_one("some text never configured")
        assert first == second

    def test_different_texts_produce_different_fallback_vectors(self):
        model = FakeEmbeddingModel()
        assert model.embed_one("text one") != model.embed_one("text two")

    def test_embed_preserves_input_order(self):
        model = FakeEmbeddingModel(vectors={"a": [1.0] * 8, "b": [2.0] * 8})
        vectors = model.embed(["a", "b"])
        assert vectors == [[1.0] * 8, [2.0] * 8]

    def test_dimensions_must_be_positive(self):
        with pytest.raises(ValueError, match="dimensions"):
            FakeEmbeddingModel(dimensions=0)

    def test_configured_vector_dimension_mismatch_is_rejected(self):
        model = FakeEmbeddingModel(vectors={"x": [1.0, 2.0]}, dimensions=8)
        with pytest.raises(ValueError, match="dimensions"):
            model.embed_one("x")

    def test_get_config_reports_model_id_and_dimensions(self):
        model = FakeEmbeddingModel(dimensions=4)
        assert model.get_config() == {"model_id": "fake-embedding-v1", "dimensions": 4}

    def test_fallback_vector_has_configured_dimensions(self):
        model = FakeEmbeddingModel(dimensions=16)
        assert len(model.embed_one("anything")) == 16
