"""Tests for RagPipeline: prompt construction, retrieval/generation separation, and metadata."""

import pytest

from llm_reliability.evaluation import ExecutionRequest, MockAdapter, ModelConfig
from llm_reliability.evaluation.adapters import AdapterError
from llm_reliability.rag.documents import Chunk, Document
from llm_reliability.rag.embeddings import FakeEmbeddingModel
from llm_reliability.rag.pipeline import (
    RagPipeline,
    RagPipelineConfig,
    build_rag_prompt,
    compute_corpus_fingerprint,
    extract_rag_metadata,
    extract_retrieved_chunks,
)
from llm_reliability.rag.retriever import RetrievedChunk, Retriever


def _config(**overrides) -> RagPipelineConfig:
    defaults = dict(
        embedding_model_id="fake-embedding-v1",
        chunk_size=100,
        chunk_overlap=0,
        top_k=2,
        corpus_fingerprint="abc123",
    )
    defaults.update(overrides)
    return RagPipelineConfig(**defaults)


def _request(
    test_case_id: str = "t1", question: str = "What is the capital of France?"
) -> ExecutionRequest:
    return ExecutionRequest(
        test_case_id=test_case_id, input_text=question, model_config=ModelConfig(model_id="m1")
    )


class TestRagPipelineConfig:
    def test_round_trip_through_dict(self):
        config = _config()
        assert RagPipelineConfig.from_dict(config.to_dict()) == config


class TestBuildRagPrompt:
    def test_includes_question_and_ranked_context(self):
        chunks = [
            RetrievedChunk(
                chunk_id="c1", document_id="doc-1", text="Paris is the capital.", score=0.9, rank=1
            )
        ]
        prompt = build_rag_prompt("What is the capital of France?", chunks)
        assert "What is the capital of France?" in prompt
        assert "Paris is the capital." in prompt
        assert "doc-1" in prompt
        assert "[1]" in prompt

    def test_handles_no_retrieved_chunks_explicitly(self):
        prompt = build_rag_prompt("A question", [])
        assert "no context retrieved" in prompt

    def test_context_order_matches_rank_order(self):
        chunks = [
            RetrievedChunk(chunk_id="c1", document_id="d1", text="first chunk", score=0.9, rank=1),
            RetrievedChunk(chunk_id="c2", document_id="d2", text="second chunk", score=0.5, rank=2),
        ]
        prompt = build_rag_prompt("q", chunks)
        assert prompt.index("first chunk") < prompt.index("second chunk")


class TestComputeCorpusFingerprint:
    def test_identical_corpus_produces_identical_fingerprint(self):
        docs = [Document(id="d1", content="hello"), Document(id="d2", content="world")]
        assert compute_corpus_fingerprint(docs) == compute_corpus_fingerprint(docs)

    def test_order_independent(self):
        a = [Document(id="d1", content="hello"), Document(id="d2", content="world")]
        b = [Document(id="d2", content="world"), Document(id="d1", content="hello")]
        assert compute_corpus_fingerprint(a) == compute_corpus_fingerprint(b)

    def test_content_change_changes_fingerprint(self):
        a = [Document(id="d1", content="hello")]
        b = [Document(id="d1", content="goodbye")]
        assert compute_corpus_fingerprint(a) != compute_corpus_fingerprint(b)


class TestRagPipeline:
    def _pipeline(self, generation_adapter=None, top_k=2):
        chunks = [
            Chunk(
                id="c-relevant",
                document_id="doc-france",
                text="Paris is the capital of France.",
                position=0,
            ),
            Chunk(
                id="c-irrelevant", document_id="doc-other", text="Bananas are yellow.", position=0
            ),
        ]
        embedding_model = FakeEmbeddingModel(
            vectors={
                "What is the capital of France?": [1, 0, 0, 0, 0, 0, 0, 0],
                "Paris is the capital of France.": [1, 0, 0, 0, 0, 0, 0, 0],
                "Bananas are yellow.": [0, 1, 0, 0, 0, 0, 0, 0],
            }
        )
        retriever = Retriever(chunks=chunks, embedding_model=embedding_model, top_k=top_k)
        adapter = generation_adapter or MockAdapter(responses={"t1": "Paris"})
        return RagPipeline(
            retriever=retriever, generation_adapter=adapter, config=_config(top_k=top_k)
        )

    def test_generate_returns_generation_adapter_output(self):
        pipeline = self._pipeline()
        response = pipeline.generate(_request())
        assert response.output_text == "Paris"

    def test_retrieval_evidence_is_attached_to_response(self):
        pipeline = self._pipeline()
        response = pipeline.generate(_request())
        chunks = extract_retrieved_chunks(response)
        assert chunks is not None
        assert chunks[0].chunk_id == "c-relevant"
        assert chunks[0].rank == 1

    def test_rag_metadata_includes_prompt_and_config(self):
        pipeline = self._pipeline()
        response = pipeline.generate(_request())
        rag_metadata = extract_rag_metadata(response)
        assert rag_metadata is not None
        assert "What is the capital of France?" in rag_metadata["prompt"]
        assert rag_metadata["pipeline_config"]["top_k"] == 2

    def test_generation_adapter_receives_constructed_prompt_not_raw_question(self):
        captured_requests = []

        class CapturingAdapter(MockAdapter):
            def generate(self, request):
                captured_requests.append(request)
                return super().generate(request)

        pipeline = self._pipeline(generation_adapter=CapturingAdapter(responses={"t1": "Paris"}))
        pipeline.generate(_request())

        assert len(captured_requests) == 1
        assert captured_requests[0].input_text != "What is the capital of France?"
        assert "Paris is the capital of France." in captured_requests[0].input_text

    def test_generation_failure_propagates(self):
        pipeline = self._pipeline(
            generation_adapter=MockAdapter(errors={"t1": "generation exploded"})
        )
        with pytest.raises(AdapterError, match="generation exploded"):
            pipeline.generate(_request())

    def test_response_with_no_rag_metadata_extracts_none(self):
        plain_response = MockAdapter(responses={"t1": "answer"}).generate(_request())
        assert extract_retrieved_chunks(plain_response) is None
        assert extract_rag_metadata(plain_response) is None

    def test_underlying_provider_metadata_is_preserved_alongside_rag_metadata(self):
        pipeline = self._pipeline()
        response = pipeline.generate(_request())
        assert response.provider_metadata["adapter"] == "mock"
        assert "rag" in response.provider_metadata


def test_concise_template_asks_for_short_answer_and_unknown_template_fails() -> None:
    from llm_reliability.rag.pipeline import build_rag_prompt as _build

    prompt = _build("Who?", [], template_id="concise_v1")
    assert "short phrase" in prompt and prompt.endswith("Answer:")
    with pytest.raises(ValueError):
        _build("Who?", [], template_id="missing")


class TestContextPerturbations:
    """RQ5 perturbations change only the prompt, never the recorded retrieval."""

    def _pipeline(self, **config_overrides):
        chunks = [
            Chunk(id="c-a", document_id="d-a", text="Paris is the capital of France.", position=0),
            Chunk(id="c-b", document_id="d-b", text="France is in Europe.", position=0),
            Chunk(id="c-z", document_id="d-z", text="Bananas are yellow.", position=0),
        ]
        embedding_model = FakeEmbeddingModel(
            vectors={
                "What is the capital of France?": [1.0, 0.2, 0.0],
                "Paris is the capital of France.": [1.0, 0.0, 0.0],
                "France is in Europe.": [0.6, 0.8, 0.0],
                "Bananas are yellow.": [0.0, 0.0, 1.0],
            },
            dimensions=3,
        )
        retriever = Retriever(chunks=chunks, embedding_model=embedding_model, top_k=2)
        echo = MockAdapter(model_id="echo")  # echoes the prompt back as output
        return RagPipeline(
            retriever=retriever,
            generation_adapter=echo,
            config=_config(top_k=2, **config_overrides),
        )

    def test_default_ranked_order_numbers_passages_by_position(self):
        prompt = self._pipeline().generate(_request()).output_text
        assert prompt.index("[1] (source: d-a)") < prompt.index("[2] (source: d-b)")

    def test_reversed_order_puts_best_passage_last_without_revealing_rank(self):
        response = self._pipeline(context_order="reversed").generate(_request())
        prompt = response.output_text
        assert "[1] (source: d-b)" in prompt and "[2] (source: d-a)" in prompt
        # Recorded retrieval is unchanged: the retriever still ranked c-a first.
        assert [c.chunk_id for c in extract_retrieved_chunks(response)] == ["c-a", "c-b"]

    def test_distractor_is_least_similar_chunk_placed_first_and_recorded_separately(self):
        response = self._pipeline(distractor_chunks=1).generate(_request())
        prompt = response.output_text
        assert prompt.index("(source: d-z)") < prompt.index("(source: d-a)")
        assert [c.chunk_id for c in extract_retrieved_chunks(response)] == ["c-a", "c-b"]
        injected = extract_rag_metadata(response)["injected_chunks"]
        assert [c["chunk_id"] for c in injected] == ["c-z"]
        assert injected[0]["rank"] == 3

    def test_default_settings_keep_the_pre_existing_config_dict(self):
        keys = set(_config().to_dict())
        assert "context_order" not in keys and "distractor_chunks" not in keys
        perturbed = _config(context_order="reversed", distractor_chunks=1)
        assert RagPipelineConfig.from_dict(perturbed.to_dict()) == perturbed

    def test_invalid_settings_are_rejected(self):
        with pytest.raises(ValueError, match="context_order"):
            _config(context_order="shuffled")
        with pytest.raises(ValueError, match="distractor_chunks"):
            _config(distractor_chunks=-1)
