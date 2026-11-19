"""
Export a deterministic multi-experiment dataset, plus derived reliability
diagnostics, robustness comparisons, and analytical summaries as JSON fixtures
for the frontend research console.

Executes controlled, reproducible scenarios through the real persistence path:
1. Primary RAG execution (Top-K=1, full diagnostics, lexical claims, robustness).
2. Repeated baseline execution (identical configuration for stability analysis).
3. Controlled intervention execution (Top-K=2 for parameter sensitivity comparison).
4. Contrast non-RAG baseline execution (differing dataset for incompatibility handling).

Nothing here is a real LLM call or a real external measurement; every fixture
file is explicitly labeled as deterministic fixture data and the frontend must
never present it as live experiment data.

Usage:
    python scripts/export_frontend_fixtures.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from llm_reliability.analytics import AnalyticsService
from llm_reliability.api.registry import GENERATION_RESPONSES, JUDGE_RESPONSES
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
from llm_reliability.rag.evaluators import (
    HitAtKEvaluator,
    MeanReciprocalRankEvaluator,
    RecallAtKEvaluator,
    evaluate_retrieval,
)
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

    # Shared with the API registry and seed script (one source of truth).
    generation_responses = GENERATION_RESPONSES
    judge_responses = JUDGE_RESPONSES

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

    with tempfile.TemporaryDirectory(prefix="llm-reliability-fixture-store-") as store_dir:
        repo = LocalFileExperimentRepository(store_dir)
        manager = ExperimentManager(repo)

        # 1. Primary RAG Experiment Definition (Top-K=1)
        definition_k1 = manager.create_experiment(
            experiment_id="exp-reliability-console-demo",
            name="Reliability console demonstration",
            dataset=dataset,
            model_config=model_config_k1,
            evaluators=evaluators,
            description=(
                "Deterministic, fully controlled RAG experiment used to generate fixture "
                "data for the frontend research console. Uses fake embeddings, the "
                "deterministic mock adapter, and a deterministic fake judge throughout; "
                "measures nothing about a real model or a real retrieval system."
            ),
        )
        # Run 1: Primary baseline execution
        run_1 = manager.execute_experiment(
            definition_k1, pipeline_k1, evaluators, adapter_id="rag-pipeline-v1"
        )
        # Run 2: Repeated baseline execution (testing stability / reproducibility)
        run_2_repeated = manager.execute_experiment(
            definition_k1, pipeline_k1, evaluators, adapter_id="rag-pipeline-v1"
        )

        # 2. Controlled Intervention RAG Experiment Definition (Top-K=2)
        definition_k2 = manager.create_experiment(
            experiment_id="exp-reliability-topk2-demo",
            name="RAG Top-K=2 Intervention",
            dataset=dataset,
            model_config=model_config_k2,
            evaluators=evaluators,
            description=(
                "Controlled intervention: top_k increased to 2 on the identical corpus and dataset."
            ),
        )
        # Run 3: Controlled intervention execution
        run_3_intervention = manager.execute_experiment(
            definition_k2, pipeline_k2, evaluators, adapter_id="rag-pipeline-topk-2"
        )

        # 3. Small non-RAG experiment for contrast (different dataset version)
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
        run_plain = manager.execute_experiment(
            plain_definition, plain_adapter, plain_evaluators, adapter_id="mock-adapter-v1"
        )

        # Reload all runs from persistence
        fresh_repo = LocalFileExperimentRepository(store_dir)
        fresh_manager = ExperimentManager(fresh_repo)

        all_runs = [
            fresh_manager.load_run(run_1.run_id),
            fresh_manager.load_run(run_2_repeated.run_id),
            fresh_manager.load_run(run_3_intervention.run_id),
            fresh_manager.load_run(run_plain.run_id),
        ]
        all_defs = [
            fresh_manager.load_experiment(definition_k1.experiment_id),
            fresh_manager.load_experiment(definition_k2.experiment_id),
            fresh_manager.load_experiment(plain_definition.experiment_id),
        ]

        # Compute diagnostics, retrieval, and robustness maps per run
        diagnostics_map: dict[str, dict[str, Any]] = {}
        retrieval_map: dict[str, dict[str, Any]] = {}
        robustness_map: dict[str, dict[str, Any]] = {}
        summaries_map: dict[str, dict[str, Any]] = {}

        ret_evaluators = [
            HitAtKEvaluator(k=1),
            RecallAtKEvaluator(k=1),
            MeanReciprocalRankEvaluator(),
        ]

        for r in all_runs:
            if r.result is None:
                continue

            r_dataset = r.dataset_version.dataset
            r_tc_by_id = {tc.id: tc for tc in r_dataset.test_cases}
            r_result_by_id = {tc_res.test_case_id: tc_res for tc_res in r.result.test_case_results}
            r_responses_by_id = {
                tc_id: tc_res.model_response
                for tc_id, tc_res in r_result_by_id.items()
                if tc_res.model_response is not None
            }

            # Retrieval results
            has_rag = any(
                tc_res.model_response and "rag" in tc_res.model_response.provider_metadata
                for tc_res in r.result.test_case_results
            )
            if has_rag:
                r_ret_results = evaluate_retrieval(
                    r_dataset.test_cases, r_responses_by_id, ret_evaluators
                )
                retrieval_map[r.run_id] = {
                    tc_id: [res.to_dict() for res in res_list]
                    for tc_id, res_list in r_ret_results.items()
                }
            else:
                r_ret_results = {}

            # Diagnostics
            r_diags = {
                tc_id: build_reliability_diagnostic(
                    r_tc_by_id[tc_id], r_result_by_id[tc_id], r_ret_results.get(tc_id, [])
                )
                for tc_id in r_tc_by_id
            }
            for d in r_diags.values():
                if d.retrieved_context and d.generated_answer:
                    d.claim_assessments = assess_statements_lexically(
                        d.generated_answer, d.retrieved_context
                    )
            diagnostics_map[r.run_id] = {tc_id: d.to_dict() for tc_id, d in r_diags.items()}

            # Robustness
            r_rob_results = []
            for tc_id, test_case in r_tc_by_id.items():
                info = get_perturbation_info(test_case.metadata)
                if info is None:
                    continue
                baseline_id, perturbation_type = info
                if baseline_id in r_diags and tc_id in r_diags:
                    r_rob_results.append(
                        compare_robustness(r_diags[baseline_id], r_diags[tc_id], perturbation_type)
                    )
            if r_rob_results:
                consistency = summarize_robustness(r_rob_results)
                robustness_map[r.run_id] = {
                    "comparisons": [rob.to_dict() for rob in r_rob_results],
                    "consistency": consistency.to_dict(),
                }

            # Evaluation & Reliability summaries
            eval_sum = summarize_run(r.result)
            rel_findings = analyze_reliability(r.result)
            ret_sum = summarize_retrieval(r_ret_results)
            summaries_map[r.run_id] = {
                "evaluation_summary": eval_sum.to_dict(),
                "reliability_findings": [f.to_dict() for f in rel_findings],
                "retrieval_summary": {name: s.to_dict() for name, s in ret_sum.items()},
            }

        # Initialize AnalyticsService for advanced cross-run analytics
        def load_diags(run_id: str):
            data = diagnostics_map.get(run_id, {})
            from llm_reliability.diagnostics.diagnosis import ReliabilityDiagnostic

            return [ReliabilityDiagnostic.from_dict(v) for v in data.values()]

        def load_retrieval(run_id: str):
            data = retrieval_map.get(run_id, {})
            from llm_reliability.rag.evaluators import RetrievalEvaluationResult

            return [
                RetrievalEvaluationResult.from_dict(res)
                for res_list in data.values()
                for res in res_list
            ]

        def load_robustness(run_id: str):
            data = robustness_map.get(run_id, {}).get("comparisons", [])
            from llm_reliability.diagnostics.robustness import RobustnessResult

            return [RobustnessResult.from_dict(res) for res in data]

        analytics_service = AnalyticsService(
            repository=fresh_repo,
            diagnostics_loader=load_diags,
            retrieval_loader=load_retrieval,
            robustness_loader=load_robustness,
        )

        overview = analytics_service.get_overview()

        # Generate comparative studies
        comparisons = {
            "identical": analytics_service.compare([run_1.run_id, run_2_repeated.run_id]).to_dict(),
            "intervention": analytics_service.compare(
                [run_1.run_id, run_3_intervention.run_id]
            ).to_dict(),
            "incompatible": analytics_service.compare([run_1.run_id, run_plain.run_id]).to_dict(),
        }

        # Generate per-run analytical summaries
        evaluator_analytics: dict[str, Any] = {}
        failure_analytics: dict[str, Any] = {}
        retrieval_analytics: dict[str, Any] = {}
        robustness_analytics: dict[str, Any] = {}
        performance_analytics: dict[str, Any] = {}

        for r in all_runs:
            evaluator_analytics[r.run_id] = {
                "aggregates": {
                    k: v.to_dict()
                    for k, v in analytics_service.get_run_evaluation_aggregates(r.run_id).items()
                },
                "agreements": [a.to_dict() for a in analytics_service.get_run_agreements(r.run_id)],
            }
            failure_analytics[r.run_id] = analytics_service.get_run_failures(r.run_id).to_dict()
            retrieval_analytics[r.run_id] = [
                s.to_dict() for s in analytics_service.get_run_retrieval(r.run_id)
            ]
            rob_sum = analytics_service.get_run_robustness(r.run_id)
            robustness_analytics[r.run_id] = rob_sum.to_dict() if rob_sum else None
            performance_analytics[r.run_id] = analytics_service.get_run_performance(
                r.run_id
            ).to_dict()

        trends = {
            definition_k1.experiment_id: analytics_service.get_trend(
                experiment_id=definition_k1.experiment_id
            ).to_dict()
        }

        # --- Write fixtures ---
        if OUTPUT_DIR.exists():
            shutil.rmtree(OUTPUT_DIR)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        def write(name: str, payload) -> None:
            (OUTPUT_DIR / name).write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )

        write("experiments.json", [d.to_dict() for d in all_defs])
        # Mirror the API's derived provenance fields so fixture mode shows the
        # same "simulated" labelling as live mode.
        write(
            "runs.json",
            {
                r.run_id: {
                    **r.to_dict(),
                    "executed_model_ids": r.executed_model_ids,
                    "simulated_components": r.simulated_components,
                }
                for r in all_runs
            },
        )
        write("diagnostics.json", diagnostics_map)
        write("retrieval_results.json", retrieval_map)
        write("robustness.json", robustness_map)
        write("summaries.json", summaries_map)
        write("analytics_overview.json", overview.to_dict())
        write("comparisons.json", comparisons)
        write("evaluator_analytics.json", evaluator_analytics)
        write("failure_analytics.json", failure_analytics)
        write("retrieval_analytics.json", retrieval_analytics)
        write("robustness_analytics.json", robustness_analytics)
        write("performance_analytics.json", performance_analytics)
        write("trends.json", trends)

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
                "experiment_ids": [d.experiment_id for d in all_defs],
                "run_ids": [r.run_id for r in all_runs],
            },
        )

    print(f"Wrote fixtures to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
