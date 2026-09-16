"""
Regression tests confirming that Prompt 2's experiment fingerprinting
mechanism (unchanged in this milestone) correctly incorporates the new
evaluators' configuration -- normalization settings, similarity threshold
and backend, and judge model/rubric -- without any change to
``llm_reliability.experiments``. The fingerprint is derived automatically
from ``EvaluatorConfig.parameters``, which is exactly ``Evaluator.get_config()``,
so a new evaluator's configuration becomes part of the reproducibility
record purely by implementing ``get_config()`` correctly.
"""

from llm_reliability.evaluation import Dataset, MockAdapter, ModelConfig, TestCase
from llm_reliability.evaluation.evaluators import ExactMatchEvaluator
from llm_reliability.evaluation.llm_judge import CORRECTNESS_RUBRIC, JudgeRubric, LLMJudgeEvaluator
from llm_reliability.evaluation.semantic_similarity import (
    FakeSimilarityBackend,
    SemanticSimilarityEvaluator,
)
from llm_reliability.experiments.models import DatasetVersion, EvaluatorConfig, ExperimentDefinition


def _dataset_version() -> DatasetVersion:
    dataset = Dataset(name="ds", test_cases=[TestCase(id="t1", input="q", reference_answer="a")])
    return DatasetVersion(dataset=dataset)


def _definition(evaluators, experiment_id: str = "exp-1") -> ExperimentDefinition:
    return ExperimentDefinition(
        experiment_id=experiment_id,
        name="fingerprint-test",
        dataset_version=_dataset_version(),
        model_config=ModelConfig(model_id="mock-adapter-v1"),
        evaluator_configs=[EvaluatorConfig.from_evaluator(e) for e in evaluators],
    )


class TestExactMatchNormalizationChangesFingerprint:
    def test_case_sensitivity_changes_fingerprint(self):
        baseline = _definition([ExactMatchEvaluator(case_sensitive=False)])
        changed = _definition([ExactMatchEvaluator(case_sensitive=True)])
        assert baseline.config_fingerprint != changed.config_fingerprint

    def test_collapse_whitespace_changes_fingerprint(self):
        baseline = _definition([ExactMatchEvaluator(collapse_whitespace=True)])
        changed = _definition([ExactMatchEvaluator(collapse_whitespace=False)])
        assert baseline.config_fingerprint != changed.config_fingerprint

    def test_identical_configuration_produces_identical_fingerprint(self):
        first = _definition([ExactMatchEvaluator(case_sensitive=True)], experiment_id="exp-a")
        second = _definition([ExactMatchEvaluator(case_sensitive=True)], experiment_id="exp-b")
        assert first.config_fingerprint == second.config_fingerprint


class TestSemanticSimilarityConfigurationChangesFingerprint:
    def test_threshold_change_changes_fingerprint(self):
        baseline = _definition(
            [SemanticSimilarityEvaluator(backend=FakeSimilarityBackend(), threshold=0.85)]
        )
        changed = _definition(
            [SemanticSimilarityEvaluator(backend=FakeSimilarityBackend(), threshold=0.5)]
        )
        assert baseline.config_fingerprint != changed.config_fingerprint

    def test_backend_model_type_change_changes_fingerprint(self):
        from llm_reliability.evaluation.semantic_similarity import BertScoreBackend

        baseline = _definition(
            [SemanticSimilarityEvaluator(backend=BertScoreBackend(model_type="model-a"))]
        )
        changed = _definition(
            [SemanticSimilarityEvaluator(backend=BertScoreBackend(model_type="model-b"))]
        )
        assert baseline.config_fingerprint != changed.config_fingerprint


class TestLLMJudgeConfigurationChangesFingerprint:
    def _judge_config(self):
        return ModelConfig(model_id="judge-v1")

    def test_judge_model_id_change_changes_fingerprint(self):
        judge = MockAdapter()
        baseline = _definition(
            [
                LLMJudgeEvaluator(
                    judge_adapter=judge, judge_model_config=ModelConfig(model_id="judge-a")
                )
            ]
        )
        changed = _definition(
            [
                LLMJudgeEvaluator(
                    judge_adapter=judge, judge_model_config=ModelConfig(model_id="judge-b")
                )
            ]
        )
        assert baseline.config_fingerprint != changed.config_fingerprint

    def test_rubric_change_changes_fingerprint(self):
        judge = MockAdapter()
        other_rubric = JudgeRubric(
            rubric_id="other_v1",
            criterion=CORRECTNESS_RUBRIC.criterion,
            scale=CORRECTNESS_RUBRIC.scale,
            instructions="A completely different set of instructions for the judge.",
        )
        baseline = _definition(
            [LLMJudgeEvaluator(judge_adapter=judge, judge_model_config=self._judge_config())]
        )
        changed = _definition(
            [
                LLMJudgeEvaluator(
                    judge_adapter=judge,
                    judge_model_config=self._judge_config(),
                    rubric=other_rubric,
                )
            ]
        )
        assert baseline.config_fingerprint != changed.config_fingerprint

    def test_identical_rubric_and_model_produces_identical_fingerprint(self):
        judge = MockAdapter()
        first = _definition(
            [LLMJudgeEvaluator(judge_adapter=judge, judge_model_config=self._judge_config())],
            experiment_id="exp-a",
        )
        second = _definition(
            [LLMJudgeEvaluator(judge_adapter=judge, judge_model_config=self._judge_config())],
            experiment_id="exp-b",
        )
        assert first.config_fingerprint == second.config_fingerprint


class TestMultiEvaluatorComposition:
    def test_adding_an_evaluator_changes_fingerprint(self):
        baseline = _definition([ExactMatchEvaluator()])
        with_semantic = _definition(
            [ExactMatchEvaluator(), SemanticSimilarityEvaluator(backend=FakeSimilarityBackend())]
        )
        assert baseline.config_fingerprint != with_semantic.config_fingerprint

    def test_evaluator_order_changes_fingerprint(self):
        exact = ExactMatchEvaluator()
        semantic = SemanticSimilarityEvaluator(backend=FakeSimilarityBackend())
        first_order = _definition([exact, semantic])
        second_order = _definition([semantic, exact])
        assert first_order.config_fingerprint != second_order.config_fingerprint
