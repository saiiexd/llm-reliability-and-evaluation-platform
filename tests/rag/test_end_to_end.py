"""
End-to-end test of the RAG execution and retrieval evaluation layer.

Covers the full pipeline required for this milestone: Experiment ->
ExperimentRun -> Retrieval -> Generation -> (retrieval evaluation, answer
evaluation) -> Persistence -> Reload, using deterministic fake components
throughout. Includes controlled cases for every failure mode this
milestone's diagnostic layer distinguishes: correct retrieval and correct
answer, missing retrieval, irrelevant retrieval, relevant retrieval with
an incorrect answer, contradictory retrieval, multiple relevant documents,
and insufficient evidence (no ground truth). This validates the
infrastructure; it makes no scientific claim about real retrieval or
generation quality.
"""

from llm_reliability.evaluation import (
    Dataset,
    ExactMatchEvaluator,
    MockAdapter,
    ModelConfig,
    TestCase,
)
from llm_reliability.experiments.local_repository import LocalFileExperimentRepository
from llm_reliability.experiments.manager import ExperimentManager
from llm_reliability.rag.context_support import ContextSupportBaseline
from llm_reliability.rag.diagnosis import RagDiagnosisCategory, diagnose_rag_execution
from llm_reliability.rag.documents import ChunkingConfig, Document, FixedSizeChunker
from llm_reliability.rag.embeddings import FakeEmbeddingModel
from llm_reliability.rag.evaluators import HitAtKEvaluator, RecallAtKEvaluator, evaluate_retrieval
from llm_reliability.rag.pipeline import RagPipeline, RagPipelineConfig, compute_corpus_fingerprint
from llm_reliability.rag.retriever import Retriever

DOCUMENTS = [
    Document(
        id="doc-france", content="Paris is the capital of France. France is in Western Europe."
    ),
    Document(id="doc-japan", content="Tokyo is the capital of Japan. Japan is an island nation."),
    Document(
        id="doc-fruit", content="Bananas are a good source of potassium and are yellow when ripe."
    ),
    Document(
        id="doc-france-contradiction",
        content="Some historical sources incorrectly claimed Lyon was the capital of France.",
    ),
]

QUESTION_FRANCE = "What is the capital of France?"
QUESTION_JAPAN = "What is the capital of Japan?"
QUESTION_UNKNOWN = "What is the capital of a country with no matching document?"


def _build_corpus():
    chunker = FixedSizeChunker(ChunkingConfig(chunk_size=200, chunk_overlap=0))
    return chunker.chunk_documents(DOCUMENTS)


def _build_embedding_model():
    # Explicit, controlled vectors: each axis represents a distinct topic,
    # so similarity is fully predictable for every scenario below.
    vectors = {
        QUESTION_FRANCE: [1, 0, 0, 0],
        QUESTION_JAPAN: [0, 1, 0, 0],
        # Deliberately closer to the japan/fruit topics than to france, so the
        # france-relevant chunk scores strictly lower than the top two and is
        # excluded from top-k -- not merely excluded by tie-break ordering.
        QUESTION_UNKNOWN: [0, 1, 1, 0],
        "Paris is the capital of France. France is in Western Europe.": [1, 0, 0, 0],
        "Tokyo is the capital of Japan. Japan is an island nation.": [0, 1, 0, 0],
        "Bananas are a good source of potassium and are yellow when ripe.": [0, 0, 1, 0],
        "Some historical sources incorrectly claimed Lyon was the capital of France.": [
            0.9,
            0,
            0,
            0,
        ],
    }
    return FakeEmbeddingModel(vectors=vectors, dimensions=4)


def _pipeline_config(top_k: int) -> RagPipelineConfig:
    return RagPipelineConfig(
        embedding_model_id="fake-embedding-v1",
        chunk_size=200,
        chunk_overlap=0,
        top_k=top_k,
        corpus_fingerprint=compute_corpus_fingerprint(DOCUMENTS),
    )


def _build_dataset() -> Dataset:
    corpus = _build_corpus()
    france_chunk_id = next(c.id for c in corpus if c.document_id == "doc-france")
    fruit_chunk_id = next(c.id for c in corpus if c.document_id == "doc-fruit")

    return Dataset(
        name="rag-e2e",
        test_cases=[
            # 1. Correct information retrieved, correct answer generated.
            TestCase(
                id="correct-retrieval-correct-answer",
                input=QUESTION_FRANCE,
                reference_answer="Paris",
                metadata={"relevant_chunk_ids": [france_chunk_id]},
            ),
            # 2. Correct information NOT retrieved (top_k too small to reach it,
            #    simulated by asking about Japan but only ever retrieving top-1
            #    from a corpus ordered so the relevant chunk ranks low). Modeled
            #    directly by declaring a relevant id that is deliberately never
            #    retrieved for this query embedding.
            TestCase(
                id="missing-retrieval",
                input=QUESTION_UNKNOWN,
                reference_answer="Unknown",
                metadata={"relevant_chunk_ids": [france_chunk_id]},
            ),
            # 3. Irrelevant information retrieved (banana chunk for a France question
            #    would only happen with a broken embedding; here we assert on the
            #    ground-truth check directly by pointing relevance at a chunk that
            #    is topically unrelated to the query embedding cluster).
            TestCase(
                id="irrelevant-retrieval-recorded",
                input=QUESTION_FRANCE,
                reference_answer="Paris",
                metadata={"relevant_chunk_ids": [fruit_chunk_id]},
            ),
            # 4. Relevant information retrieved but answer incorrect.
            TestCase(
                id="relevant-retrieval-wrong-answer",
                input=QUESTION_FRANCE,
                reference_answer="Paris",
                metadata={"relevant_chunk_ids": [france_chunk_id]},
            ),
            # 5. Contradictory information retrieved (a second, low-similarity
            #    document contradicts the correct one); ground truth still points
            #    at the correct chunk.
            TestCase(
                id="contradictory-retrieval",
                input=QUESTION_FRANCE,
                reference_answer="Paris",
                metadata={"relevant_chunk_ids": [france_chunk_id]},
            ),
            # 6. Multiple relevant documents/chunks retrieved.
            TestCase(
                id="multiple-relevant",
                input=QUESTION_FRANCE,
                reference_answer="Paris",
                metadata={"relevant_chunk_ids": [france_chunk_id, fruit_chunk_id]},
            ),
            # 7. Insufficient evidence: no retrieval ground truth at all.
            TestCase(id="insufficient-evidence", input=QUESTION_JAPAN, reference_answer="Tokyo"),
        ],
    )


def _build_generation_responses():
    return {
        "correct-retrieval-correct-answer": "Paris",
        "missing-retrieval": "I do not know.",
        "irrelevant-retrieval-recorded": "Paris",
        "relevant-retrieval-wrong-answer": "Berlin",
        "contradictory-retrieval": "Paris",
        "multiple-relevant": "Paris",
        "insufficient-evidence": "Tokyo",
    }


def test_rag_experiment_through_persistence_and_reload_with_diagnosis(tmp_path):
    corpus = _build_corpus()
    embedding_model = _build_embedding_model()
    dataset = _build_dataset()

    retriever = Retriever(chunks=corpus, embedding_model=embedding_model, top_k=2)
    generation_adapter = MockAdapter(
        responses=_build_generation_responses(), model_id="mock-adapter-v1"
    )
    pipeline = RagPipeline(
        retriever=retriever, generation_adapter=generation_adapter, config=_pipeline_config(top_k=2)
    )

    exact_match = ExactMatchEvaluator()
    context_support = ContextSupportBaseline(min_overlap_ratio=0.2)
    evaluators = [exact_match, context_support]

    model_config = ModelConfig(
        model_id="mock-adapter-v1", extra_params={"rag": _pipeline_config(top_k=2).to_dict()}
    )

    manager = ExperimentManager(LocalFileExperimentRepository(tmp_path))
    definition = manager.create_experiment(
        experiment_id="exp-rag-e2e",
        name="RAG execution and retrieval evaluation demonstration",
        dataset=dataset,
        model_config=model_config,
        evaluators=evaluators,
    )
    run = manager.execute_experiment(definition, pipeline, evaluators)

    assert run.result.num_test_cases == 7
    assert run.result.num_succeeded == 7  # generation always succeeds; correctness varies

    # Reload from persistence, as a fresh process would.
    fresh_manager = ExperimentManager(LocalFileExperimentRepository(tmp_path))
    reloaded_run = fresh_manager.load_run(run.run_id)
    reloaded_dataset = reloaded_run.dataset_version.dataset

    by_id = {r.test_case_id: r for r in reloaded_run.result.test_case_results}
    responses_by_id = {tc_id: tc_result.model_response for tc_id, tc_result in by_id.items()}

    retrieval_evaluators = [HitAtKEvaluator(k=2), RecallAtKEvaluator(k=2)]
    retrieval_results = evaluate_retrieval(
        reloaded_dataset.test_cases, responses_by_id, retrieval_evaluators
    )

    # 1. Correct retrieval, correct answer -> successful grounded execution.
    diagnosis_1 = diagnose_rag_execution(
        "correct-retrieval-correct-answer",
        retrieval_results["correct-retrieval-correct-answer"],
        by_id["correct-retrieval-correct-answer"].evaluation_results,
    )
    assert diagnosis_1.category == RagDiagnosisCategory.SUCCESSFUL_GROUNDED_EXECUTION

    # 2. Missing retrieval (relevant chunk never surfaces for this query) and a
    #    wrong answer -> evidence consistent with retrieval failure.
    diagnosis_2 = diagnose_rag_execution(
        "missing-retrieval",
        retrieval_results["missing-retrieval"],
        by_id["missing-retrieval"].evaluation_results,
    )
    assert diagnosis_2.category == RagDiagnosisCategory.RETRIEVAL_FAILURE

    # 4. Relevant chunk retrieved but the answer is wrong -> generation failure.
    diagnosis_4 = diagnose_rag_execution(
        "relevant-retrieval-wrong-answer",
        retrieval_results["relevant-retrieval-wrong-answer"],
        by_id["relevant-retrieval-wrong-answer"].evaluation_results,
    )
    assert diagnosis_4.category == RagDiagnosisCategory.GENERATION_FAILURE_DESPITE_RELEVANT_CONTEXT

    # 6. Multiple relevant targets: recall should reflect partial coverage
    #    when only one of two relevant chunks is retrievable in top-k.
    multiple_relevant_recall = next(
        r for r in retrieval_results["multiple-relevant"] if r.metric_name == "recall_at_k"
    )
    assert multiple_relevant_recall.score is not None
    assert 0.0 < multiple_relevant_recall.score < 1.0

    # 7. Insufficient evidence: no ground truth was provided at all.
    diagnosis_7 = diagnose_rag_execution(
        "insufficient-evidence",
        retrieval_results.get("insufficient-evidence", []),
        by_id["insufficient-evidence"].evaluation_results,
    )
    assert diagnosis_7.category == RagDiagnosisCategory.RETRIEVAL_EVIDENCE_UNAVAILABLE

    # 3. Irrelevant retrieval: ground truth points at a chunk from a different
    #    topic than what was actually retrieved, so the hit signal is False,
    #    yet the answer happens to be correct anyway. That combination is
    #    genuinely ambiguous given only these two signals, so it must be
    #    reported as undetermined, never silently treated as success.
    irrelevant_hit = next(
        r for r in retrieval_results["irrelevant-retrieval-recorded"] if r.metric_name == "hit_at_k"
    )
    assert irrelevant_hit.score == 0.0
    diagnosis_3 = diagnose_rag_execution(
        "irrelevant-retrieval-recorded",
        retrieval_results["irrelevant-retrieval-recorded"],
        by_id["irrelevant-retrieval-recorded"].evaluation_results,
    )
    assert diagnosis_3.category == RagDiagnosisCategory.UNDETERMINED

    # 5. Contradictory retrieval: both the correct chunk and a contradicting
    #    one are retrieved; ground truth still identifies the correct chunk,
    #    so the hit signal is True, and the answer is correct.
    contradictory_hit = next(
        r for r in retrieval_results["contradictory-retrieval"] if r.metric_name == "hit_at_k"
    )
    assert contradictory_hit.score == 1.0
    contradictory_chunk_ids = {
        rc["chunk_id"]
        for rc in by_id["contradictory-retrieval"].model_response.provider_metadata["rag"][
            "retrieved_chunks"
        ]
    }
    assert "doc-france-contradiction::chunk-0" in contradictory_chunk_ids
    diagnosis_5 = diagnose_rag_execution(
        "contradictory-retrieval",
        retrieval_results["contradictory-retrieval"],
        by_id["contradictory-retrieval"].evaluation_results,
    )
    assert diagnosis_5.category == RagDiagnosisCategory.SUCCESSFUL_GROUNDED_EXECUTION

    # An incorrect answer alone (test case 4) must never be classified as
    # retrieval failure when the relevant chunk actually was retrieved.
    assert diagnosis_4.category != RagDiagnosisCategory.RETRIEVAL_FAILURE

    # Retrieved chunks, ranks, scores, and configuration are preserved exactly
    # through persistence and reload.
    rag_metadata = by_id["correct-retrieval-correct-answer"].model_response.provider_metadata["rag"]
    assert rag_metadata["retrieval_config"]["top_k"] == 2
    assert (
        rag_metadata["pipeline_config"]["corpus_fingerprint"]
        == _pipeline_config(top_k=2).corpus_fingerprint
    )
    assert len(rag_metadata["retrieved_chunks"]) == 2
    assert rag_metadata["retrieved_chunks"][0]["rank"] == 1

    # Configuration fingerprint is preserved and reflects the RAG configuration
    # embedded in ModelConfig.extra_params -- no changes to the experiment
    # system's fingerprinting were required for this.
    assert reloaded_run.config_fingerprint == run.config_fingerprint
    assert reloaded_run.model_config.extra_params["rag"]["top_k"] == 2
