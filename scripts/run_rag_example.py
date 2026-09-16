"""
Minimal demonstration of the RAG execution and retrieval evaluation layer.

Builds a tiny document corpus, chunks it deterministically, retrieves with
a deterministic fake embedding model, generates with the deterministic
mock adapter, evaluates the answer and the retrieval separately, and
produces an evidence-based RAG failure diagnosis for each test case --
covering a correct retrieval/correct answer case, a missing-retrieval
case, and a relevant-retrieval/wrong-answer case. Persists the experiment
and run, reloads them, and re-derives the same diagnoses from the
reloaded data.

This is an infrastructure demonstration, not a scientific measurement:
all retrieval and generation behavior here is deterministic and
hand-configured, using no real embedding model or LLM.

Requires no network access, no API credentials, and no external services.
Persisted files are written to a temporary directory removed on exit.

Usage:
    python scripts/run_rag_example.py
"""

from __future__ import annotations

import sys
import tempfile

from llm_reliability.evaluation import (
    Dataset,
    ExactMatchEvaluator,
    MockAdapter,
    ModelConfig,
    TestCase,
)
from llm_reliability.experiments import ExperimentManager, LocalFileExperimentRepository
from llm_reliability.rag.diagnosis import diagnose_rag_execution
from llm_reliability.rag.documents import ChunkingConfig, Document, FixedSizeChunker
from llm_reliability.rag.embeddings import FakeEmbeddingModel
from llm_reliability.rag.evaluators import HitAtKEvaluator, evaluate_retrieval
from llm_reliability.rag.pipeline import RagPipeline, RagPipelineConfig, compute_corpus_fingerprint
from llm_reliability.rag.retriever import Retriever

DOCUMENTS = [
    Document(id="doc-france", content="Paris is the capital of France."),
    Document(id="doc-japan", content="Tokyo is the capital of Japan."),
]


def build_corpus_and_embeddings():
    chunker = FixedSizeChunker(ChunkingConfig(chunk_size=200, chunk_overlap=0))
    chunks = chunker.chunk_documents(DOCUMENTS)
    embedding_model = FakeEmbeddingModel(
        vectors={
            "What is the capital of France?": [1, 0, 0],
            "Paris is the capital of France.": [1, 0, 0],
            # Deliberately identical to the japan chunk's vector, so with
            # top_k=1 the japan chunk is retrieved and the france-relevant
            # chunk is excluded -- not merely excluded by tie-break order.
            "What is the capital of a country not in this corpus?": [0, 1, 0],
            "Tokyo is the capital of Japan.": [0, 1, 0],
        },
        dimensions=3,
    )
    return chunks, embedding_model


def build_dataset(chunks):
    france_chunk_id = next(c.id for c in chunks if c.document_id == "doc-france")
    return Dataset(
        name="rag-example",
        test_cases=[
            TestCase(
                id="correct-case",
                input="What is the capital of France?",
                reference_answer="Paris",
                metadata={"relevant_chunk_ids": [france_chunk_id]},
            ),
            TestCase(
                id="missing-retrieval-case",
                input="What is the capital of a country not in this corpus?",
                reference_answer="Unknown",
                metadata={"relevant_chunk_ids": [france_chunk_id]},
            ),
            TestCase(
                id="generation-failure-case",
                input="What is the capital of France?",
                reference_answer="Paris",
                metadata={"relevant_chunk_ids": [france_chunk_id]},
            ),
        ],
    )


def main() -> int:
    chunks, embedding_model = build_corpus_and_embeddings()
    dataset = build_dataset(chunks)

    retriever = Retriever(chunks=chunks, embedding_model=embedding_model, top_k=1)
    generation_adapter = MockAdapter(
        responses={
            "correct-case": "Paris",
            "missing-retrieval-case": "I do not know.",
            "generation-failure-case": "Berlin",  # wrong, despite relevant context
        },
        model_id="mock-adapter-v1",
    )
    pipeline_config = RagPipelineConfig(
        embedding_model_id=embedding_model.model_id,
        chunk_size=200,
        chunk_overlap=0,
        top_k=1,
        corpus_fingerprint=compute_corpus_fingerprint(DOCUMENTS),
    )
    pipeline = RagPipeline(
        retriever=retriever, generation_adapter=generation_adapter, config=pipeline_config
    )

    exact_match = ExactMatchEvaluator()
    model_config = ModelConfig(
        model_id="mock-adapter-v1", extra_params={"rag": pipeline_config.to_dict()}
    )

    with tempfile.TemporaryDirectory(prefix="llm-reliability-rag-store-") as store_dir:
        print(f"Local experiment store: {store_dir}\n")
        manager = ExperimentManager(LocalFileExperimentRepository(store_dir))

        definition = manager.create_experiment(
            experiment_id="exp-rag-demo",
            name="RAG execution demonstration",
            dataset=dataset,
            model_config=model_config,
            evaluators=[exact_match],
            description=(
                "Infrastructure demonstration of RAG execution and diagnosis; not a real "
                "measurement."
            ),
        )
        run = manager.execute_experiment(definition, pipeline, [exact_match])
        print(
            f"Run: {run.run_id}  status={run.status}  config_fingerprint={run.config_fingerprint}\n"
        )

        fresh_manager = ExperimentManager(LocalFileExperimentRepository(store_dir))
        reloaded_run = fresh_manager.load_run(run.run_id)
        assert reloaded_run.result is not None, "run must have produced a result to demonstrate"
        reloaded_dataset = reloaded_run.dataset_version.dataset

        by_id = {r.test_case_id: r for r in reloaded_run.result.test_case_results}
        responses_by_id = {
            tc_id: tc_result.model_response
            for tc_id, tc_result in by_id.items()
            if tc_result.model_response is not None
        }
        retrieval_results = evaluate_retrieval(
            reloaded_dataset.test_cases, responses_by_id, [HitAtKEvaluator(k=1)]
        )

        print("--- Per-test-case diagnosis (from reloaded data) ---")
        for test_case in reloaded_dataset.test_cases:
            tc_result = by_id[test_case.id]
            assert tc_result.model_response is not None
            diagnosis = diagnose_rag_execution(
                test_case.id, retrieval_results.get(test_case.id, []), tc_result.evaluation_results
            )
            print(f"[{test_case.id}] answer={tc_result.model_response.output_text!r}")
            print(f"  diagnosis: {diagnosis.category.value}")
            print(f"  evidence: {diagnosis.evidence}")
            print(f"  explanation: {diagnosis.explanation}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
