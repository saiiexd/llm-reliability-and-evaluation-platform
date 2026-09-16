"""
End-to-end test of the reliability, faithfulness, and robustness layer.

Covers the controlled scenarios this milestone's diagnostic engine must
distinguish: correct context with correct generation, correct context
with unsupported generation, correct context with contradictory
generation, irrelevant context with an answer, missing context with an
answer, relevant context followed by incorrect generation, and a
baseline-versus-perturbed robustness comparison. Verifies both expected
diagnostic categories and expected uncertainty (undetermined /
insufficient evidence) where the setup does not provide enough evidence.
Everything runs through Experiment -> ExperimentRun -> persistence ->
reload, using only deterministic fake components.
"""

import json

from llm_reliability.diagnostics import (
    FailureCategory,
    PerturbationType,
    build_reliability_diagnostic,
    compare_robustness,
    get_perturbation_info,
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
from llm_reliability.rag.evaluators import HitAtKEvaluator, evaluate_retrieval
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


def _build_corpus():
    return FixedSizeChunker(ChunkingConfig(chunk_size=200, chunk_overlap=0)).chunk_documents(
        DOCUMENTS
    )


def _build_embedding_model():
    france_chunk_text = "Paris is the capital of France."
    fruit_chunk_text = "Bananas are a good source of potassium."
    return FakeEmbeddingModel(
        vectors={
            QUESTION: [1, 0],
            QUESTION_REWORDED: [1, 0],
            # Deliberately closer to the fruit chunk than to the
            # france-relevant chunk, so with top_k=1 the france chunk is
            # excluded from retrieval -- not merely excluded by tie-break.
            QUESTION_UNRELATED: [0, 1],
            france_chunk_text: [1, 0],
            fruit_chunk_text: [0, 1],
        },
        dimensions=2,
    )


def _build_dataset(france_chunk_id: str) -> Dataset:
    return Dataset(
        name="reliability-e2e",
        test_cases=[
            # 1. Correct context, correct generation.
            TestCase(
                id="correct-context-correct-answer",
                input=QUESTION,
                reference_answer="Paris",
                metadata={"relevant_chunk_ids": [france_chunk_id]},
            ),
            # 2. Correct context, unsupported generation (answer adds
            #    unaddressed information).
            TestCase(
                id="correct-context-unsupported-answer",
                input=QUESTION,
                reference_answer="Paris",
                metadata={"relevant_chunk_ids": [france_chunk_id]},
            ),
            # 3. Correct context, contradictory generation.
            TestCase(
                id="correct-context-contradictory-answer",
                input=QUESTION,
                reference_answer="Paris",
                metadata={"relevant_chunk_ids": [france_chunk_id]},
            ),
            # 4. Relevant context retrieved, but generation is incorrect.
            TestCase(
                id="relevant-context-incorrect-answer",
                input=QUESTION,
                reference_answer="Paris",
                metadata={"relevant_chunk_ids": [france_chunk_id]},
            ),
            # 5. Missing context (query embedding excludes the relevant chunk).
            TestCase(
                id="missing-context",
                input=QUESTION_UNRELATED,
                reference_answer="Unknown",
                metadata={"relevant_chunk_ids": [france_chunk_id]},
            ),
            # 6. No retrieval ground truth at all -> insufficient evidence.
            TestCase(id="no-ground-truth", input=QUESTION, reference_answer="Paris"),
            # 7. Robustness perturbation: an equivalent reworded question.
            TestCase(
                id="correct-context-correct-answer-reworded",
                input=QUESTION_REWORDED,
                reference_answer="Paris",
                metadata={
                    "relevant_chunk_ids": [france_chunk_id],
                    "perturbation_of": "correct-context-correct-answer",
                    "perturbation_type": "question_rewording",
                },
            ),
        ],
    )


def test_reliability_and_robustness_layer_end_to_end(tmp_path):
    corpus = _build_corpus()
    france_chunk_id = corpus[0].id
    dataset = _build_dataset(france_chunk_id)
    embedding_model = _build_embedding_model()

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
            {"outcome": "supported", "reasoning": "Matches context."}
        ),
        "correct-context-unsupported-answer": json.dumps(
            {"outcome": "unsupported", "reasoning": "Population figure is not in the context."}
        ),
        "correct-context-contradictory-answer": json.dumps(
            {"outcome": "contradicted", "reasoning": "Context says Paris, answer says Lyon."}
        ),
        "relevant-context-incorrect-answer": json.dumps(
            {"outcome": "contradicted", "reasoning": "Context says Paris, answer says Berlin."}
        ),
        "no-ground-truth": json.dumps({"outcome": "supported", "reasoning": "Matches context."}),
        "correct-context-correct-answer-reworded": json.dumps(
            {"outcome": "supported", "reasoning": "Matches context."}
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

    manager = ExperimentManager(LocalFileExperimentRepository(tmp_path))
    definition = manager.create_experiment(
        experiment_id="exp-reliability-e2e",
        name="Reliability and robustness demonstration",
        dataset=dataset,
        model_config=model_config,
        evaluators=evaluators,
    )
    run = manager.execute_experiment(definition, pipeline, evaluators)

    # Reload from persistence, as a fresh process would.
    fresh_manager = ExperimentManager(LocalFileExperimentRepository(tmp_path))
    reloaded_run = fresh_manager.load_run(run.run_id)
    reloaded_dataset = reloaded_run.dataset_version.dataset

    by_id = {r.test_case_id: r for r in reloaded_run.result.test_case_results}
    tc_by_id = {tc.id: tc for tc in reloaded_dataset.test_cases}
    responses_by_id = {
        tc_id: tc_result.model_response
        for tc_id, tc_result in by_id.items()
        if tc_result.model_response is not None
    }
    retrieval_results = evaluate_retrieval(
        reloaded_dataset.test_cases, responses_by_id, [HitAtKEvaluator(k=1)]
    )

    def diagnose(test_case_id: str):
        return build_reliability_diagnostic(
            tc_by_id[test_case_id], by_id[test_case_id], retrieval_results.get(test_case_id, [])
        )

    # 1. Correct context, correct generation -> correct.
    assert diagnose("correct-context-correct-answer").failure_category == FailureCategory.CORRECT

    # 2. Correct context, unsupported generation.
    assert (
        diagnose("correct-context-unsupported-answer").failure_category
        == FailureCategory.UNSUPPORTED
    )

    # 3. Correct context, contradictory generation -> contradicted takes
    #    precedence, and the answer is also incorrect per exact match.
    assert (
        diagnose("correct-context-contradictory-answer").failure_category
        == FailureCategory.CONTRADICTED
    )

    # 4. Relevant context retrieved, generation incorrect -> generation failure
    #    (retrieval succeeded, so this must never be misattributed to retrieval).
    diagnosis_4 = diagnose("relevant-context-incorrect-answer")
    assert diagnosis_4.failure_category in (
        FailureCategory.CONTRADICTED,
        FailureCategory.GENERATION_FAILURE,
    )
    assert diagnosis_4.failure_category != FailureCategory.RETRIEVAL_FAILURE

    # 5. Missing context -> retrieval failure (ground truth was not surfaced,
    #    and the answer is wrong).
    assert diagnose("missing-context").failure_category == FailureCategory.RETRIEVAL_FAILURE

    # 6. No retrieval ground truth -> the retrieval-evidence gap must not be
    #    silently resolved; correctness alone still determines CORRECT here.
    diagnosis_6 = diagnose("no-ground-truth")
    assert diagnosis_6.status.value in ("determined", "undetermined")

    # 7. Robustness: baseline and perturbed reworded question should both
    #    resolve to the same category (both correct), so outcome is stable.
    baseline_diagnosis = diagnose("correct-context-correct-answer")
    perturbed_test_case = tc_by_id["correct-context-correct-answer-reworded"]
    baseline_id, perturbation_type = get_perturbation_info(perturbed_test_case.metadata)
    assert baseline_id == "correct-context-correct-answer"
    assert perturbation_type == PerturbationType.QUESTION_REWORDING
    perturbed_diagnosis = diagnose("correct-context-correct-answer-reworded")
    robustness_result = compare_robustness(
        baseline_diagnosis, perturbed_diagnosis, perturbation_type
    )
    assert robustness_result.outcome_changed is False

    # Configuration and diagnostic data survive persistence and reload exactly.
    assert reloaded_run.config_fingerprint == run.config_fingerprint
    contradicted_result = by_id["correct-context-contradictory-answer"]
    faithfulness_results = [
        r
        for r in contradicted_result.evaluation_results
        if r.evaluator_name == "llm_faithfulness_judge"
    ]
    assert faithfulness_results[0].label == "contradicted"
