"""
Opt-in integration test for the real Sentence Transformers embedding backend.

This test is NOT part of the standard offline test suite. It requires
both 'sentence-transformers' to be importable AND the
LLM_RELIABILITY_RUN_INTEGRATION_TESTS environment variable to be set to
"1", so that it never runs merely because the package happens to be
present in a shared environment. Even when the package is installed, the
first real call downloads and caches a pretrained model, which requires
network access; this is exactly why the behavioral tests in
test_embeddings.py and test_retriever.py use FakeEmbeddingModel instead.

Run explicitly with (CPU-only torch first, to avoid the CUDA wheels):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install ".[embeddings]"
    LLM_RELIABILITY_RUN_INTEGRATION_TESTS=1 pytest tests/rag/test_embeddings_integration.py -v

These tests check structural correctness and directionally expected
behavior (semantically related sentences should be more similar than
unrelated ones; real retrieval over the bundled SQuAD subset should find
the source passage far more often than chance). They do not assert exact
embedding values, which depend on the installed model version.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

from llm_reliability.datasets.squad import load_squad_subset
from llm_reliability.rag.embeddings import SentenceTransformerEmbeddingModel
from llm_reliability.rag.retriever import Retriever
from llm_reliability.rag.similarity import cosine_similarity

_SENTENCE_TRANSFORMERS_AVAILABLE = importlib.util.find_spec("sentence_transformers") is not None
_INTEGRATION_TESTS_ENABLED = os.environ.get("LLM_RELIABILITY_RUN_INTEGRATION_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not (_SENTENCE_TRANSFORMERS_AVAILABLE and _INTEGRATION_TESTS_ENABLED),
    reason=(
        "opt-in integration test: requires sentence-transformers installed and "
        "LLM_RELIABILITY_RUN_INTEGRATION_TESTS=1"
    ),
)


def test_real_embeddings_rank_related_text_above_unrelated_text():
    query = "What is the capital of France?"
    related = "Paris is the capital city of France."
    unrelated = "The mitochondria is the powerhouse of the cell."

    model = SentenceTransformerEmbeddingModel()
    query_vector, related_vector, unrelated_vector = model.embed([query, related, unrelated])

    related_score = cosine_similarity(query_vector, related_vector)
    unrelated_score = cosine_similarity(query_vector, unrelated_vector)

    assert related_score > unrelated_score


def test_real_embeddings_through_retriever_on_the_squad_subset():
    """End-to-end real retrieval: all 60 questions against the full 378-chunk corpus.

    Same corpus and chunking as EXP-001. A question is a hit when one of its
    derived ``relevant_chunk_ids`` is in the top 3. With 378 chunks, a random
    ranking would hit roughly 1 question in 60; the bar below (more than a
    third of questions) was fixed before the test was first run and is a
    sanity floor, not a quality target.
    """
    subset = load_squad_subset()
    chunks = subset.chunks()
    dataset = subset.dataset()
    model = SentenceTransformerEmbeddingModel()
    retriever = Retriever(chunks=chunks, embedding_model=model, top_k=3)

    assert retriever.config.embedding_model_id == "sentence-transformers/all-MiniLM-L6-v2"
    chunk_ids = {chunk.id for chunk in chunks}
    hits = 0
    for test_case in dataset.test_cases:
        result = retriever.retrieve(test_case.input)
        retrieved = result.retrieved_chunks
        assert [rc.rank for rc in retrieved] == [1, 2, 3]
        assert all(rc.chunk_id in chunk_ids for rc in retrieved)
        scores = [rc.score for rc in retrieved]
        assert scores == sorted(scores, reverse=True)
        assert all(-1.0 <= score <= 1.0 for score in scores)
        relevant = set(test_case.metadata["relevant_chunk_ids"])
        hits += any(rc.chunk_id in relevant for rc in retrieved)

    assert hits > len(dataset.test_cases) / 3
