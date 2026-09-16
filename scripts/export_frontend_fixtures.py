"""
Export a deterministic experiment run, plus its derived reliability
diagnostics and a robustness comparison, as JSON fixtures for the
frontend research console.

This script executes the same kind of controlled, deterministic scenario
used in tests/diagnostics/test_end_to_end.py (fake embeddings, the
deterministic mock adapter, and a deterministic fake judge) through the
real Experiment/ExperimentRun persistence path, then also runs a second
experiment (a small dataset with no RAG pipeline, and one with an
evaluator failure) so the frontend has representative data for every view
it needs to render. Nothing here is a real LLM call or a real measurement
of anything; every fixture file is explicitly labeled as such and the
frontend must never present it as live experiment data.

Usage:
    python scripts/export_frontend_fixtures.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

from llm_reliability.diagnostics import (
    assess_statements_lexically,
    build_reliability_diagnostic,
    compare_robustness,
    get_perturbation_info,
    summarize_robustness,
)
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
from llm_reliability.rag.evaluators import HitAtKEvaluator, RecallAtKEvaluator, evaluate_retrieval
from llm_reliability.rag.faithfulness_judge import LLMFaithfulnessEvaluator
from llm_reliability.rag.pipeline import RagPipeline, RagPipelineConfig, compute_corpus_fingerprint
from llm_reliability.rag.retriever import Retriever
from llm_reliability.rag.summary import summarize_retrieval
from llm_reliability.reliability import analyze_reliability, summarize_run

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "frontend" / "src" / "data" / "fixtures"

DOCUMENTS = [
    Document(id="doc-france", content="Paris is the capital of France."),
    Document(id="doc-fruit", content="Bananas are a good source of potassium."),
]
QUESTION = "What is the capital of France?"
QUESTION_REWORDED = "Which city serves as the capital of France?"
QUESTION_UNRELATED = "What is the capital of a country not in this corpus?"


def build_corpus():
    return FixedSizeChunker(ChunkingConfig(chunk_size=200, chunk_overlap=0)).chunk_documents(
        DOCUMENTS
    )


def build_embedding_model():
    france_text = "Paris is the capital of France."
    fruit_text = "Bananas are a good source of potassium."
    return FakeEmbeddingModel(
        vectors={
            QUESTION: [1, 0],
            QUESTION_REWORDED: [1, 0],
            QUESTION_UNRELATED: [0, 1],
            france_text: [1, 0],
            fruit_text: [0, 1],
        },
        dimensions=2,
    )


def build_dataset(france_chunk_id: str) -> Dataset:
    return Dataset(
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
                metadata={
                    "relevant_chunk_ids": [france_chunk_id],
                    "category": "generation_failure",
                },
            ),
            TestCase(
                id="missing-context",
                input=QUESTION_UNRELATED,
                reference_answer="Unknown",
                metadata={"relevant_chunk_ids": [france_chunk_id], "category": "retrieval_failure"},
            ),
            TestCase(
                id="no-ground-truth",
                input=QUESTION,
                reference_answer="Paris",
                metadata={"category": "insufficient_evidence"},
            ),
            TestCase(
                id="correct-context-correct-answer-reworded",
                input=QUESTION_REWORDED,
                reference_answer="Paris",
                metadata={
                    "relevant_chunk_ids": [france_chunk_id],
                    "perturbation_of": "correct-context-correct-answer",
                    "perturbation_type": "question_rewording",
                    "category": "robustness",
                },
            ),
        ],
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    corpus = build_corpus()
    france_chunk_id = corpus[0].id
    dataset = build_dataset(france_chunk_id)
    embedding_model = build_embedding_model()

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
            {"outcome": "supported", "reasoning": "Matches the retrieved context exactly."}
        ),
        "correct-context-unsupported-answer": json.dumps(
            {
                "outcome": "unsupported",
                "reasoning": "The population figure is not present in the retrieved context.",
            }
        ),
        "correct-context-contradictory-answer": json.dumps(
            {
                "outcome": "contradicted",
                "reasoning": "The context states Paris; the answer states Lyon.",
            }
        ),
        "relevant-context-incorrect-answer": json.dumps(
            {
                "outcome": "contradicted",
                "reasoning": "The context states Paris; the answer states Berlin.",
            }
        ),
        "no-ground-truth": json.dumps(
            {"outcome": "supported", "reasoning": "Matches the retrieved context."}
        ),
        "correct-context-correct-answer-reworded": json.dumps(
            {"outcome": "supported", "reasoning": "Matches the retrieved context."}
        ),
    }

    retriever = Retriever(chunks=corpus, embedding_model=embedding_model, top_k=1)
    generation_adapter = MockAdapter(responses=generation_responses, model_id="mock-adapter-v1")
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

    judge_adapter = MockAdapter(responses=judge_responses, model_id="fake-judge-v1")
    exact_match = ExactMatchEvaluator()
    faithfulness_judge = LLMFaithfulnessEvaluator(
        judge_adapter=judge_adapter, judge_model_config=ModelConfig(model_id="fake-judge-v1")
    )
    evaluators = [exact_match, faithfulness_judge]
    model_config = ModelConfig(
        model_id="mock-adapter-v1", extra_params={"rag": pipeline_config.to_dict()}
    )

    with tempfile.TemporaryDirectory(prefix="llm-reliability-fixture-store-") as store_dir:
        manager = ExperimentManager(LocalFileExperimentRepository(store_dir))
        definition = manager.create_experiment(
            experiment_id="exp-reliability-console-demo",
            name="Reliability console demonstration",
            dataset=dataset,
            model_config=model_config,
            evaluators=evaluators,
            description=(
                "Deterministic, fully controlled RAG experiment used to generate fixture "
                "data for the frontend research console. Uses fake embeddings, the "
                "deterministic mock adapter, and a deterministic fake judge throughout; "
                "measures nothing about a real model or a real retrieval system."
            ),
        )
        run = manager.execute_experiment(definition, pipeline, evaluators)

        # Second, small non-RAG experiment for contrast (no retrieval evidence at all).
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
            experiment_id="exp-baseline-non-rag-demo",
            name="Baseline non-RAG demonstration",
            dataset=plain_dataset,
            model_config=ModelConfig(model_id="mock-adapter-v1"),
            evaluators=plain_evaluators,
            description="A plain (non-RAG) experiment, for contrast with the RAG demonstration.",
        )
        plain_run = manager.execute_experiment(plain_definition, plain_adapter, plain_evaluators)

        # Reload everything from persistence, exactly as the frontend's data
        # access boundary should eventually do against a real backend.
        fresh_manager = ExperimentManager(LocalFileExperimentRepository(store_dir))
        reloaded_definition = fresh_manager.load_experiment(definition.experiment_id)
        reloaded_run = fresh_manager.load_run(run.run_id)
        reloaded_plain_definition = fresh_manager.load_experiment(plain_definition.experiment_id)
        reloaded_plain_run = fresh_manager.load_run(plain_run.run_id)
        assert reloaded_run.result is not None
        assert reloaded_plain_run.result is not None

        reloaded_dataset = reloaded_run.dataset_version.dataset
        tc_by_id = {tc.id: tc for tc in reloaded_dataset.test_cases}
        result_by_id = {r.test_case_id: r for r in reloaded_run.result.test_case_results}
        responses_by_id = {
            tc_id: r.model_response
            for tc_id, r in result_by_id.items()
            if r.model_response is not None
        }
        retrieval_results = evaluate_retrieval(
            reloaded_dataset.test_cases,
            responses_by_id,
            [HitAtKEvaluator(k=1), RecallAtKEvaluator(k=1)],
        )

        diagnostics = {
            tc_id: build_reliability_diagnostic(
                tc_by_id[tc_id], result_by_id[tc_id], retrieval_results.get(tc_id, [])
            )
            for tc_id in tc_by_id
        }
        # Attach statement-level lexical evidence assessments wherever
        # retrieved context exists, so the evidence-inspection view has
        # real claim-level data to render (see llm_reliability.diagnostics.evidence).
        for diagnostic in diagnostics.values():
            if diagnostic.retrieved_context and diagnostic.generated_answer:
                diagnostic.claim_assessments = assess_statements_lexically(
                    diagnostic.generated_answer, diagnostic.retrieved_context
                )

        robustness_results = []
        for tc_id, test_case in tc_by_id.items():
            info = get_perturbation_info(test_case.metadata)
            if info is None:
                continue
            baseline_id, perturbation_type = info
            robustness_results.append(
                compare_robustness(diagnostics[baseline_id], diagnostics[tc_id], perturbation_type)
            )
        consistency = summarize_robustness(robustness_results)

        evaluation_summary = summarize_run(reloaded_run.result)
        reliability_findings = analyze_reliability(reloaded_run.result)
        retrieval_summary = summarize_retrieval(retrieval_results)

        plain_evaluation_summary = summarize_run(reloaded_plain_run.result)

        # --- Write fixtures ---
        if OUTPUT_DIR.exists():
            shutil.rmtree(OUTPUT_DIR)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        def write(name: str, payload) -> None:
            (OUTPUT_DIR / name).write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )

        write(
            "experiments.json", [reloaded_definition.to_dict(), reloaded_plain_definition.to_dict()]
        )
        write(
            "runs.json",
            {
                reloaded_run.run_id: reloaded_run.to_dict(),
                reloaded_plain_run.run_id: reloaded_plain_run.to_dict(),
            },
        )
        write(
            "diagnostics.json",
            {reloaded_run.run_id: {tc_id: d.to_dict() for tc_id, d in diagnostics.items()}},
        )
        write(
            "retrieval_results.json",
            {
                reloaded_run.run_id: {
                    tc_id: [r.to_dict() for r in results]
                    for tc_id, results in retrieval_results.items()
                }
            },
        )
        write(
            "robustness.json",
            {
                reloaded_run.run_id: {
                    "comparisons": [r.to_dict() for r in robustness_results],
                    "consistency": consistency.to_dict(),
                }
            },
        )
        write(
            "summaries.json",
            {
                reloaded_run.run_id: {
                    "evaluation_summary": evaluation_summary.to_dict(),
                    "reliability_findings": [f.to_dict() for f in reliability_findings],
                    "retrieval_summary": {
                        name: s.to_dict() for name, s in retrieval_summary.items()
                    },
                },
                reloaded_plain_run.run_id: {
                    "evaluation_summary": plain_evaluation_summary.to_dict(),
                    "reliability_findings": [],
                    "retrieval_summary": {},
                },
            },
        )
        write(
            "meta.json",
            {
                "generated_by": "scripts/export_frontend_fixtures.py",
                "is_fixture_data": True,
                "warning": (
                    "This data was generated by a deterministic, fully controlled "
                    "script using fake embeddings and a fake judge. It is not the "
                    "output of a real LLM, a real retrieval system, or a real "
                    "experiment, and must never be presented as live experiment data."
                ),
                "experiment_ids": [definition.experiment_id, plain_definition.experiment_id],
                "run_ids": [run.run_id, plain_run.run_id],
            },
        )

    print(f"Wrote fixtures to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
