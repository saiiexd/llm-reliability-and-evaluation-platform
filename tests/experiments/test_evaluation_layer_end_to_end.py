"""
End-to-end test of the Evaluation and Reliability Layer.

Covers the full pipeline required for this milestone: an experiment
configured with multiple evaluators (exact match, semantic similarity, and
LLM-as-a-judge) executed once via the existing EvaluationRunner, with
per-test-case, per-evaluator results preserved through persistence and
reload, evaluator disagreement observable in the reloaded data, and a
reliability/summary analysis computed from the reloaded results.
"""

import json

from llm_reliability.evaluation import (
    Dataset,
    ExactMatchEvaluator,
    MockAdapter,
    ModelConfig,
    TestCase,
)
from llm_reliability.evaluation.llm_judge import LLMJudgeEvaluator
from llm_reliability.evaluation.semantic_similarity import (
    FakeSimilarityBackend,
    SemanticSimilarityEvaluator,
)
from llm_reliability.experiments.local_repository import LocalFileExperimentRepository
from llm_reliability.experiments.manager import ExperimentManager
from llm_reliability.reliability import ReliabilityFlagType, analyze_reliability, summarize_run


class _CountingAdapter(MockAdapter):
    """Wraps MockAdapter to count generate() calls, for verifying call counts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.call_count = 0

    def generate(self, request):
        self.call_count += 1
        return super().generate(request)


def test_system_under_test_is_called_exactly_once_per_test_case_regardless_of_evaluator_count(
    tmp_path,
):
    dataset = Dataset(
        name="call-count-check",
        test_cases=[
            TestCase(id="t1", input="q1", reference_answer="a1"),
            TestCase(id="t2", input="q2", reference_answer="a2"),
        ],
    )
    adapter = _CountingAdapter(responses={"t1": "a1", "t2": "a2"})
    judge_adapter = MockAdapter(
        responses={
            "t1": json.dumps({"outcome": "correct", "reasoning": "ok"}),
            "t2": json.dumps({"outcome": "correct", "reasoning": "ok"}),
        }
    )
    evaluators = [
        ExactMatchEvaluator(),
        SemanticSimilarityEvaluator(backend=FakeSimilarityBackend()),
        LLMJudgeEvaluator(
            judge_adapter=judge_adapter, judge_model_config=ModelConfig(model_id="fake-judge")
        ),
    ]

    manager = ExperimentManager(LocalFileExperimentRepository(tmp_path))
    definition = manager.create_experiment(
        experiment_id="exp-call-count",
        name="Call count check",
        dataset=dataset,
        model_config=ModelConfig(model_id="mock-adapter-v1"),
        evaluators=evaluators,
    )
    manager.execute_experiment(definition, adapter, evaluators)

    # Three evaluators are configured, but the system-under-test adapter is
    # called exactly once per test case; the judge adapter is separate and
    # is called once per test case only because LLMJudgeEvaluator explicitly
    # requires its own model call.
    assert adapter.call_count == 2


def test_multi_evaluator_experiment_persists_and_reloads_with_observable_disagreement(tmp_path):
    reference = "Paris is the capital of France."
    paraphrase = "The capital city of France is Paris."

    dataset = Dataset(
        name="evaluation-layer-e2e",
        test_cases=[
            TestCase(id="exact-agree", input="q1", reference_answer=reference),
            TestCase(id="disagreement-case", input="q2", reference_answer=reference),
            TestCase(id="no-reference-case", input="q3", reference_answer=None),
        ],
    )

    # "disagreement-case" gets a paraphrased answer: exact match will call it
    # a mismatch (different wording), while the fake similarity backend is
    # configured to report it as highly similar -- an intentional disagreement.
    adapter = MockAdapter(
        responses={
            "exact-agree": reference,
            "disagreement-case": paraphrase,
            "no-reference-case": "Some open-ended answer.",
        }
    )
    similarity_backend = FakeSimilarityBackend(similarities={(paraphrase, reference): 0.94})
    judge_adapter = MockAdapter(
        responses={
            "exact-agree": json.dumps({"outcome": "correct", "reasoning": "Matches exactly."}),
            "disagreement-case": json.dumps(
                {"outcome": "correct", "reasoning": "Same fact, reworded."}
            ),
        }
    )

    exact_match = ExactMatchEvaluator()
    semantic_similarity = SemanticSimilarityEvaluator(backend=similarity_backend, threshold=0.85)
    llm_judge = LLMJudgeEvaluator(
        judge_adapter=judge_adapter, judge_model_config=ModelConfig(model_id="fake-judge")
    )
    evaluators = [exact_match, semantic_similarity, llm_judge]

    manager = ExperimentManager(LocalFileExperimentRepository(tmp_path))
    definition = manager.create_experiment(
        experiment_id="exp-evaluation-layer",
        name="Evaluation layer demonstration",
        dataset=dataset,
        model_config=ModelConfig(model_id="mock-adapter-v1"),
        evaluators=evaluators,
    )
    run = manager.execute_experiment(definition, adapter, evaluators)

    assert run.result.num_test_cases == 3
    assert run.result.num_succeeded == 3  # generation succeeded for all three

    # Reload from persistence, as a fresh process would.
    fresh_manager = ExperimentManager(LocalFileExperimentRepository(tmp_path))
    reloaded_run = fresh_manager.load_run(run.run_id)

    by_id = {r.test_case_id: r for r in reloaded_run.result.test_case_results}

    # Agreement case: all three evaluators concur the answer is correct.
    agree_results = {r.evaluator_name: r for r in by_id["exact-agree"].evaluation_results}
    assert agree_results["exact_match"].passed is True
    assert agree_results["semantic_similarity"].passed is True
    assert agree_results["llm_judge"].passed is True

    # Disagreement case: exact match says no, semantic similarity says yes.
    disagreement_results = {
        r.evaluator_name: r for r in by_id["disagreement-case"].evaluation_results
    }
    assert disagreement_results["exact_match"].passed is False
    assert disagreement_results["semantic_similarity"].passed is True
    assert disagreement_results["llm_judge"].passed is True

    # No-reference case: reference-based evaluators skip; the judge also skips.
    no_reference_results = {
        r.evaluator_name: r for r in by_id["no-reference-case"].evaluation_results
    }
    for evaluator_name in ("exact_match", "semantic_similarity", "llm_judge"):
        assert no_reference_results[evaluator_name].status.value == "skipped"

    # Reliability analysis surfaces the disagreement without calling it a hallucination.
    findings = analyze_reliability(reloaded_run.result)
    disagreement_findings = [
        f for f in findings if f.flag_type == ReliabilityFlagType.EVALUATOR_DISAGREEMENT
    ]
    assert len(disagreement_findings) == 1
    assert disagreement_findings[0].test_case_id == "disagreement-case"
    # A finding may explicitly disclaim being a hallucination determination
    # (see ReliabilityFlagType.INCORRECT_PER_EVALUATOR), but must never
    # assert that something IS one.
    assert all("is a hallucination" not in f.detail.lower() for f in findings)

    # Evaluation summary keeps evaluators distinct; no single overall score exists.
    summary = summarize_run(reloaded_run.result)
    assert set(summary.per_evaluator.keys()) == {"exact_match", "semantic_similarity", "llm_judge"}
    assert summary.per_evaluator["exact_match"].num_skipped == 1
    assert summary.per_evaluator["exact_match"].num_passed == 1
    assert summary.per_evaluator["exact_match"].num_failed == 1
    assert summary.per_evaluator["semantic_similarity"].num_passed == 2
    assert "overall_score" not in summary.to_dict()

    # Configuration is preserved exactly through persistence and reload.
    assert reloaded_run.config_fingerprint == run.config_fingerprint
    configured_names = [ec.evaluator_name for ec in reloaded_run.evaluator_configs]
    assert configured_names == ["exact_match", "semantic_similarity", "llm_judge"]
