"""
Seed the local experiment persistence store with reproducible benchmark experiments.

Populates the repository (default: ./experiments) with:
1. exp-reliability-console-demo: Primary RAG execution (Top-K=1)
2. exp-reliability-console-demo (Run 2): Repeated baseline execution for stability
3. exp-reliability-topk2-demo: Controlled intervention (Top-K=2)
4. exp-baseline-non-rag-demo: Plain non-RAG baseline contrast

Usage:
    python scripts/seed_live_experiments.py [optional_persistence_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from llm_reliability.evaluation import (
    Dataset,
    ExactMatchEvaluator,
    MockAdapter,
    ModelConfig,
    TestCase,
)
from llm_reliability.experiments import ExperimentManager, LocalFileExperimentRepository
from llm_reliability.rag.documents import ChunkingConfig, Document, FixedSizeChunker
from llm_reliability.rag.embeddings import FakeEmbeddingModel
from llm_reliability.rag.faithfulness_judge import LLMFaithfulnessEvaluator
from llm_reliability.rag.pipeline import RagPipeline, RagPipelineConfig, compute_corpus_fingerprint
from llm_reliability.rag.retriever import Retriever

DOCUMENTS = [
    Document(id="doc-france", content="Paris is the capital of France."),
    Document(id="doc-fruit", content="Bananas are a good source of potassium."),
]
QUESTION = "What is the capital of France?"
QUESTION_REWORDED = "Which city serves as the capital of France?"
QUESTION_UNRELATED = "What is the capital of a country not in this corpus?"


def seed_store(store_path: Path) -> None:
    store_path.mkdir(parents=True, exist_ok=True)
    repo = LocalFileExperimentRepository(store_path)
    manager = ExperimentManager(repo)

    corpus = FixedSizeChunker(ChunkingConfig(chunk_size=200, chunk_overlap=0)).chunk_documents(
        DOCUMENTS
    )
    france_chunk_id = corpus[0].id

    dataset = Dataset(
        name="reliability-console-demo",
        test_cases=[
            TestCase(
                id="correct-context-correct-answer",
                input=QUESTION,
                reference_answer="Paris",
                metadata={"relevant_chunk_ids": [france_chunk_id], "category": "correct"},
            ),
            TestCase(
                id="correct-context-unsupported-answer",
                input=QUESTION,
                reference_answer="Paris",
                metadata={"relevant_chunk_ids": [france_chunk_id], "category": "unsupported"},
            ),
            TestCase(
                id="correct-context-contradictory-answer",
                input=QUESTION,
                reference_answer="Paris",
                metadata={"relevant_chunk_ids": [france_chunk_id], "category": "contradicted"},
            ),
            TestCase(
                id="relevant-context-incorrect-answer",
                input=QUESTION,
                reference_answer="Paris",
                metadata={"relevant_chunk_ids": [france_chunk_id], "category": "incorrect"},
            ),
            TestCase(
                id="missing-context",
                input=QUESTION_UNRELATED,
                reference_answer="Unavailable",
                metadata={
                    "relevant_chunk_ids": ["doc-nonexistent-chunk"],
                    "category": "retrieval_failure",
                },
            ),
            TestCase(
                id="no-ground-truth",
                input=QUESTION,
                reference_answer="Paris",
                metadata={"category": "no_ground_truth"},
            ),
            TestCase(
                id="correct-context-correct-answer-reworded",
                input=QUESTION_REWORDED,
                reference_answer="Paris",
                metadata={
                    "perturbation_of": "correct-context-correct-answer",
                    "perturbation_type": "question_rewording",
                    "relevant_chunk_ids": [france_chunk_id],
                },
            ),
        ],
    )

    france_text = "Paris is the capital of France."
    fruit_text = "Bananas are a good source of potassium."
    embedding_model = FakeEmbeddingModel(
        vectors={
            QUESTION: [1, 0],
            QUESTION_REWORDED: [1, 0],
            QUESTION_UNRELATED: [0, 1],
            france_text: [1, 0],
            fruit_text: [0, 1],
        },
        dimensions=2,
    )

    generation_responses = {
        "correct-context-correct-answer": "Paris",
        "correct-context-unsupported-answer": "Paris, which has a population of 30 million.",
        "correct-context-contradictory-answer": "Lyon is the capital of France.",
        "relevant-context-incorrect-answer": "Berlin",
        "missing-context": "I do not know.",
        "no-ground-truth": "Paris",
        "correct-context-correct-answer-reworded": "Paris",
    }
    judge_responses = {
        "correct-context-correct-answer": json.dumps(
            {"verdict": "supported", "confidence": 1.0, "explanation": "Directly supported."}
        ),
        "correct-context-unsupported-answer": json.dumps(
            {
                "verdict": "unsupported",
                "confidence": 0.9,
                "explanation": "Context does not mention population.",
            }
        ),
        "correct-context-contradictory-answer": json.dumps(
            {
                "verdict": "contradicted",
                "confidence": 1.0,
                "explanation": "Context says Paris, answer says Lyon.",
            }
        ),
        "relevant-context-incorrect-answer": json.dumps(
            {"verdict": "unsupported", "confidence": 0.8, "explanation": "Berlin is not Paris."}
        ),
        "missing-context": json.dumps(
            {"verdict": "unsupported", "confidence": 0.5, "explanation": "No relevant context."}
        ),
        "no-ground-truth": json.dumps(
            {"verdict": "supported", "confidence": 1.0, "explanation": "Supported by context."}
        ),
        "correct-context-correct-answer-reworded": json.dumps(
            {"verdict": "supported", "confidence": 1.0, "explanation": "Directly supported."}
        ),
    }

    retriever_k1 = Retriever(
        chunks=corpus,
        embedding_model=embedding_model,
        top_k=1,
    )
    pipeline_k1_config = RagPipelineConfig(
        embedding_model_id="fake-embedding-v1",
        chunk_size=200,
        chunk_overlap=0,
        top_k=1,
        corpus_fingerprint=compute_corpus_fingerprint(DOCUMENTS),
    )
    pipeline_k1 = RagPipeline(
        retriever=retriever_k1,
        generation_adapter=MockAdapter(
            responses=generation_responses, model_id="fake-rag-generator"
        ),
        config=pipeline_k1_config,
    )

    retriever_k2 = Retriever(
        chunks=corpus,
        embedding_model=embedding_model,
        top_k=2,
    )
    pipeline_k2_config = RagPipelineConfig(
        embedding_model_id="fake-embedding-v1",
        chunk_size=200,
        chunk_overlap=0,
        top_k=2,
        corpus_fingerprint=compute_corpus_fingerprint(DOCUMENTS),
    )
    pipeline_k2 = RagPipeline(
        retriever=retriever_k2,
        generation_adapter=MockAdapter(
            responses=generation_responses, model_id="fake-rag-generator"
        ),
        config=pipeline_k2_config,
    )

    judge_adapter = MockAdapter(responses=judge_responses, model_id="fake-judge-v1")
    exact_match = ExactMatchEvaluator()
    faithfulness_judge = LLMFaithfulnessEvaluator(
        judge_adapter=judge_adapter, judge_model_config=ModelConfig(model_id="fake-judge-v1")
    )
    evaluators = [exact_match, faithfulness_judge]

    model_config_k1 = ModelConfig(
        model_id="mock-adapter-v1", extra_params={"rag": pipeline_k1_config.to_dict()}
    )
    model_config_k2 = ModelConfig(
        model_id="mock-adapter-v1", extra_params={"rag": pipeline_k2_config.to_dict()}
    )

    # 1. Primary RAG Experiment Definition (Top-K=1)
    exp1_id = "exp-reliability-console-demo"
    if exp1_id not in repo.list_experiment_ids():
        definition_k1 = manager.create_experiment(
            experiment_id=exp1_id,
            name="Reliability console demonstration",
            dataset=dataset,
            model_config=model_config_k1,
            evaluators=evaluators,
            description=(
                "Deterministic, fully controlled RAG experiment used to benchmark "
                "context retrieval, faithfulness, and reliability diagnostics."
            ),
        )
    else:
        definition_k1 = manager.load_experiment(exp1_id)

    if not repo.list_run_ids(exp1_id):
        manager.execute_experiment(
            definition_k1, pipeline_k1, evaluators, run_id="run-rag-baseline-001"
        )
        manager.execute_experiment(
            definition_k1, pipeline_k1, evaluators, run_id="run-rag-baseline-002"
        )

    # 2. Controlled Intervention RAG Experiment Definition (Top-K=2)
    exp2_id = "exp-reliability-topk2-demo"
    if exp2_id not in repo.list_experiment_ids():
        definition_k2 = manager.create_experiment(
            experiment_id=exp2_id,
            name="RAG Top-K=2 Intervention",
            dataset=dataset,
            model_config=model_config_k2,
            evaluators=evaluators,
            description=(
                "Controlled intervention: top_k increased to 2 on the identical corpus and dataset."
            ),
        )
    else:
        definition_k2 = manager.load_experiment(exp2_id)

    if not repo.list_run_ids(exp2_id):
        manager.execute_experiment(
            definition_k2, pipeline_k2, evaluators, run_id="run-rag-topk2-001"
        )

    # 3. Small non-RAG experiment for contrast (different dataset version)
    exp3_id = "exp-baseline-non-rag-demo"
    if exp3_id not in repo.list_experiment_ids():
        plain_dataset = Dataset(
            name="baseline-non-rag-demo",
            test_cases=[
                TestCase(id="plain-1", input="What is 2+2?", reference_answer="4"),
                TestCase(id="plain-2", input="What is 3+3?", reference_answer="6"),
            ],
        )
        plain_adapter = MockAdapter(
            responses={"plain-1": "4", "plain-2": "7"}, model_id="mock-adapter-v1"
        )
        plain_evaluators = [ExactMatchEvaluator()]
        plain_definition = manager.create_experiment(
            experiment_id=exp3_id,
            name="Baseline non-RAG demonstration",
            dataset=plain_dataset,
            model_config=ModelConfig(model_id="mock-adapter-v1"),
            evaluators=plain_evaluators,
            description="A plain (non-RAG) experiment, for contrast with the RAG demonstration.",
        )
    else:
        plain_definition = manager.load_experiment(exp3_id)
        plain_adapter = MockAdapter(
            responses={"plain-1": "4", "plain-2": "7"}, model_id="mock-adapter-v1"
        )
        plain_evaluators = [ExactMatchEvaluator()]

    if not repo.list_run_ids(exp3_id):
        manager.execute_experiment(
            plain_definition, plain_adapter, plain_evaluators, run_id="run-non-rag-001"
        )

    print(f"Successfully seeded experiment repository at: {store_path.resolve()}")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments")
    seed_store(target)
