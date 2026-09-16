"""
Minimal demonstration of the Evaluation and Reliability Layer.

Loads the curated evaluation-methodology benchmark
(data/eval_sets/evaluation_methodology_baseline_v1.json), executes it once
through the deterministic MockAdapter, and evaluates each generated
response with three independent methods -- exact match, semantic
similarity (via a deterministic fake backend, not real BERTScore), and
LLM-as-a-judge (via MockAdapter used as a fake judge) -- then persists the
experiment and run, reloads them, and prints an evaluation summary and
reliability findings.

The mock responses below are hand-crafted to intentionally exercise every
outcome this milestone's architecture defines: exact-match/semantic-
similarity agreement, an explicit disagreement between them, a clearly
incorrect answer, an invalid judge response, a judge adapter failure, and
the reference-free "skipped" path. None of this measures anything about
real model quality; it exists only to validate the evaluation
infrastructure. Do not read any conclusion from these numbers about how
good a real model is.

Requires no network access, no API credentials, and no external services.
Persisted files are written to a temporary directory removed on exit.

Usage:
    python scripts/run_evaluation_layer_example.py
"""

from __future__ import annotations

import json
import sys
import tempfile

from llm_reliability.evaluation import (
    ExactMatchEvaluator,
    MockAdapter,
    ModelConfig,
    load_dataset_from_json,
)
from llm_reliability.evaluation.llm_judge import LLMJudgeEvaluator
from llm_reliability.evaluation.semantic_similarity import (
    FakeSimilarityBackend,
    SemanticSimilarityEvaluator,
)
from llm_reliability.experiments import ExperimentManager, LocalFileExperimentRepository
from llm_reliability.reliability import analyze_reliability, summarize_run

DATASET_PATH = "data/eval_sets/evaluation_methodology_baseline_v1.json"


def _judge_json(outcome: str, reasoning: str) -> str:
    return json.dumps({"outcome": outcome, "reasoning": reasoning})


def build_responses() -> dict[str, str]:
    """Hand-crafted mock responses. See module docstring for what each demonstrates."""
    return {
        # Exact factual: mostly correct, two deliberately wrong for contrast.
        "fact-001": "Paris",
        "fact-002": "Au",
        "fact-003": "7",
        "fact-004": "100",
        "fact-005": "William Shakespeare",
        "fact-006": "9",  # deliberately wrong (correct answer is 8)
        "fact-007": "Jupiter",
        "fact-008": "Yen",
        "fact-009": "1944",  # deliberately wrong (correct answer is 1945)
        "fact-010": "0",
        # Paraphrasable: a real paraphrase (disagreement case), a partial
        # overlap, and a fluent-but-unrelated answer, plus straightforward
        # near-identical answers for the rest.
        "para-001": (
            "Plants convert sunlight, carbon dioxide, and water into "
            "glucose and oxygen through photosynthesis."
        ),
        "para-002": "Because the atmosphere scatters blue light more than other colors.",
        "para-003": "It refers to money losing purchasing power as prices rise generally.",
        "para-004": "A firewall filters network traffic according to security rules.",
        "para-005": "It removes toxins from blood.",  # partial overlap only
        "para-006": (
            "Machine learning is a good hobby for beginners interested in technology."
        ),  # fluent but does not answer the question
        "para-007": "Gravity attracts masses toward each other, including toward Earth's center.",
        "para-008": "It tells the reader what the essay's main claim is.",
        # Ambiguous / insufficient reference.
        "amb-001": "Python is a great general-purpose language.",
        "amb-002": "Germany",  # valid answer, but not the single reference "France"
        "amb-003": "Reading is a good hobby.",
        "amb-004": "I like autumn because of the cooler weather.",
        "amb-005": "Cat",  # valid answer, but not the single reference "Dog"
        "amb-006": "I cannot determine the current time without external context.",
    }


def build_similarity_backend(dataset, responses: dict[str, str]) -> FakeSimilarityBackend:
    """Configure the fake semantic similarity backend for the paraphrase cases.

    Every score here is an arbitrary, explicitly chosen number for this
    demonstration, not a measurement -- FakeSimilarityBackend performs no
    real embedding computation. See test_semantic_similarity.py and
    test_semantic_similarity_integration.py for the evaluator's real tests.
    """
    reference_by_id = {tc.id: tc.reference_answer for tc in dataset.test_cases}
    similarities = {
        # paraphrase: disagrees with exact match
        (responses["para-001"], reference_by_id["para-001"]): 0.93,
        # partial overlap: below threshold
        (responses["para-005"], reference_by_id["para-005"]): 0.55,
        # unrelated despite fluent wording
        (responses["para-006"], reference_by_id["para-006"]): 0.05,
    }
    return FakeSimilarityBackend(similarities=similarities)


def build_judge_adapter() -> MockAdapter:
    """A deterministic fake judge: MockAdapter used as the judge model.

    Includes one deliberately invalid response (fact-006, to demonstrate
    OUTPUT_VALIDATION_ERROR) and one simulated judge outage (fact-009, to
    demonstrate EXECUTION_ERROR), alongside otherwise-consistent verdicts.
    """
    judge_responses = {}
    for test_case_id in [
        "fact-001",
        "fact-002",
        "fact-003",
        "fact-004",
        "fact-005",
        "fact-007",
        "fact-008",
        "fact-010",
    ]:
        judge_responses[test_case_id] = _judge_json("correct", "Matches the reference exactly.")
    # Invalid: not JSON per the rubric's required output format.
    judge_responses["fact-006"] = "The candidate answer looks plausible."
    judge_responses["para-001"] = _judge_json("correct", "Same fact, different wording.")
    judge_responses["para-005"] = _judge_json(
        "partially_correct", "Mentions filtering but omits digestion."
    )
    judge_responses["para-006"] = _judge_json("incorrect", "Does not answer the question asked.")
    judge_responses["amb-002"] = _judge_json(
        "partially_correct", "Germany is a valid European country, but not the reference answer."
    )
    judge_responses["amb-005"] = _judge_json(
        "partially_correct", "Cat is a valid mammal, but not the reference answer."
    )
    for test_case_id in ["para-002", "para-003", "para-004", "para-007", "para-008"]:
        judge_responses[test_case_id] = _judge_json(
            "correct", "Captures the same key fact as the reference."
        )

    return MockAdapter(
        responses=judge_responses,
        errors={"fact-009": "simulated judge model outage"},
        model_id="fake-judge-v1",
    )


def main() -> int:
    dataset = load_dataset_from_json(DATASET_PATH)
    responses = build_responses()
    adapter = MockAdapter(responses=responses, model_id="mock-adapter-v1")

    exact_match = ExactMatchEvaluator()
    semantic_similarity = SemanticSimilarityEvaluator(
        backend=build_similarity_backend(dataset, responses), threshold=0.85
    )
    judge_model_config = ModelConfig(model_id="fake-judge-v1")
    llm_judge = LLMJudgeEvaluator(
        judge_adapter=build_judge_adapter(), judge_model_config=judge_model_config
    )
    evaluators = [exact_match, semantic_similarity, llm_judge]

    with tempfile.TemporaryDirectory(prefix="llm-reliability-eval-layer-store-") as store_dir:
        print(f"Local experiment store: {store_dir}\n")
        manager = ExperimentManager(LocalFileExperimentRepository(store_dir))

        definition = manager.create_experiment(
            experiment_id="exp-evaluation-layer-demo",
            name="Evaluation layer demonstration",
            dataset=dataset,
            model_config=ModelConfig(model_id="mock-adapter-v1"),
            evaluators=evaluators,
            description=(
                "Demonstrates exact match, semantic similarity, and LLM-as-a-judge "
                "evaluating the same responses, including an intentional disagreement "
                "between exact match and semantic similarity. Uses the deterministic "
                "mock adapter throughout; measures nothing about real model quality."
            ),
        )
        print(f"Experiment: {definition.experiment_id}")
        print(f"Config fingerprint: {definition.config_fingerprint}\n")

        run = manager.execute_experiment(definition, adapter, evaluators)
        print(f"Run: {run.run_id}  status={run.status}\n")

        fresh_manager = ExperimentManager(LocalFileExperimentRepository(store_dir))
        reloaded_run = fresh_manager.load_run(run.run_id)
        assert reloaded_run.result is not None, "run must have produced a result to demonstrate"

        print("--- Disagreement case (para-001): paraphrase vs. exact match ---")
        by_id = {r.test_case_id: r for r in reloaded_run.result.test_case_results}
        for evaluator_result in by_id["para-001"].evaluation_results:
            print(
                f"  {evaluator_result.evaluator_name}: status={evaluator_result.status} "
                f"passed={evaluator_result.passed} label={evaluator_result.label}"
            )

        print("\n--- Evaluation summary (per evaluator, never a single overall score) ---")
        summary = summarize_run(reloaded_run.result)
        for evaluator_name, stats in summary.to_dict()["per_evaluator"].items():
            print(f"  {evaluator_name}: {stats}")

        print("\n--- Reliability findings ---")
        findings = analyze_reliability(reloaded_run.result)
        for finding in findings:
            print(f"  [{finding.flag_type.value}] {finding.test_case_id}: {finding.detail}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
