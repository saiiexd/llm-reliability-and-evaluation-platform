"""Tests for the Experiment System domain models."""

import pytest

from llm_reliability.evaluation import Dataset, ModelConfig, TestCase
from llm_reliability.experiments.models import (
    DatasetVersion,
    EvaluatorConfig,
    ExperimentDefinition,
    ExperimentRun,
    RunStatus,
)


def _dataset(name: str = "ds") -> Dataset:
    return Dataset(name=name, test_cases=[TestCase(id="t1", input="q", reference_answer="a")])


def _model_config() -> ModelConfig:
    return ModelConfig(model_id="mock-adapter-v1")


def _evaluator_configs() -> list[EvaluatorConfig]:
    return [EvaluatorConfig(evaluator_name="exact_match")]


class TestDatasetVersion:
    def test_version_id_is_independent_of_dataset_name(self):
        version_a = DatasetVersion(dataset=_dataset(name="name-a"))
        version_b = DatasetVersion(dataset=_dataset(name="name-b"))
        assert version_a.version_id == version_b.version_id
        assert version_a.dataset_name != version_b.dataset_name

    def test_different_test_case_content_changes_version_id(self):
        version_a = DatasetVersion(dataset=_dataset())
        different_dataset = Dataset(
            name="ds", test_cases=[TestCase(id="t1", input="different question")]
        )
        version_b = DatasetVersion(dataset=different_dataset)
        assert version_a.version_id != version_b.version_id

    def test_round_trip_through_dict(self):
        version = DatasetVersion(dataset=_dataset())
        reloaded = DatasetVersion.from_dict(version.to_dict())
        assert reloaded.version_id == version.version_id
        assert reloaded.dataset_name == version.dataset_name
        assert [tc.id for tc in reloaded.dataset.test_cases] == [
            tc.id for tc in version.dataset.test_cases
        ]

    def test_corrupted_version_id_is_rejected_on_load(self):
        version = DatasetVersion(dataset=_dataset())
        data = version.to_dict()
        data["version_id"] = "tampered"
        with pytest.raises(ValueError, match="integrity check failed"):
            DatasetVersion.from_dict(data)

    def test_tampered_test_case_content_is_detected_on_load(self):
        version = DatasetVersion(dataset=_dataset())
        data = version.to_dict()
        # version_id left as originally computed, but content is changed underneath it.
        data["test_cases"][0]["input"] = "a completely different question"
        with pytest.raises(ValueError, match="integrity check failed"):
            DatasetVersion.from_dict(data)


class TestEvaluatorConfig:
    def test_requires_non_empty_name(self):
        with pytest.raises(ValueError, match="evaluator_name"):
            EvaluatorConfig(evaluator_name="")

    def test_round_trip_through_dict(self):
        config = EvaluatorConfig(evaluator_name="exact_match", parameters={"k": "v"})
        reloaded = EvaluatorConfig.from_dict(config.to_dict())
        assert reloaded == config


class TestExperimentDefinition:
    def test_requires_non_empty_experiment_id(self):
        with pytest.raises(ValueError, match="experiment_id"):
            ExperimentDefinition(
                experiment_id="",
                name="exp",
                dataset_version=DatasetVersion(dataset=_dataset()),
                model_config=_model_config(),
                evaluator_configs=_evaluator_configs(),
            )

    def test_requires_at_least_one_evaluator_config(self):
        with pytest.raises(ValueError, match="at least one evaluator"):
            ExperimentDefinition(
                experiment_id="exp-1",
                name="exp",
                dataset_version=DatasetVersion(dataset=_dataset()),
                model_config=_model_config(),
                evaluator_configs=[],
            )

    def test_equivalent_configuration_produces_identical_fingerprint(self):
        first = ExperimentDefinition(
            experiment_id="exp-1",
            name="First",
            dataset_version=DatasetVersion(dataset=_dataset()),
            model_config=_model_config(),
            evaluator_configs=_evaluator_configs(),
        )
        second = ExperimentDefinition(
            experiment_id="exp-2",  # different identity...
            name="Second, totally different description",  # ...and metadata...
            dataset_version=DatasetVersion(dataset=_dataset()),
            model_config=_model_config(),
            evaluator_configs=_evaluator_configs(),
        )
        # ...but identical reproducibility-relevant configuration.
        assert first.config_fingerprint == second.config_fingerprint

    def test_different_model_config_produces_different_fingerprint(self):
        first = ExperimentDefinition(
            experiment_id="exp-1",
            name="exp",
            dataset_version=DatasetVersion(dataset=_dataset()),
            model_config=ModelConfig(model_id="model-a"),
            evaluator_configs=_evaluator_configs(),
        )
        second = ExperimentDefinition(
            experiment_id="exp-1",
            name="exp",
            dataset_version=DatasetVersion(dataset=_dataset()),
            model_config=ModelConfig(model_id="model-b"),
            evaluator_configs=_evaluator_configs(),
        )
        assert first.config_fingerprint != second.config_fingerprint

    def test_different_dataset_content_produces_different_fingerprint(self):
        first = ExperimentDefinition(
            experiment_id="exp-1",
            name="exp",
            dataset_version=DatasetVersion(dataset=_dataset()),
            model_config=_model_config(),
            evaluator_configs=_evaluator_configs(),
        )
        different_dataset = Dataset(name="ds", test_cases=[TestCase(id="t1", input="other")])
        second = ExperimentDefinition(
            experiment_id="exp-1",
            name="exp",
            dataset_version=DatasetVersion(dataset=different_dataset),
            model_config=_model_config(),
            evaluator_configs=_evaluator_configs(),
        )
        assert first.config_fingerprint != second.config_fingerprint

    def test_round_trip_through_json(self):
        definition = ExperimentDefinition(
            experiment_id="exp-1",
            name="exp",
            dataset_version=DatasetVersion(dataset=_dataset()),
            model_config=_model_config(),
            evaluator_configs=_evaluator_configs(),
            description="A test experiment.",
            metadata={"owner": "research-team"},
        )
        reloaded = ExperimentDefinition.from_json(definition.to_json())
        assert reloaded.experiment_id == definition.experiment_id
        assert reloaded.config_fingerprint == definition.config_fingerprint
        assert reloaded.metadata == definition.metadata

    def test_tampered_fingerprint_is_rejected_on_load(self):
        definition = ExperimentDefinition(
            experiment_id="exp-1",
            name="exp",
            dataset_version=DatasetVersion(dataset=_dataset()),
            model_config=_model_config(),
            evaluator_configs=_evaluator_configs(),
        )
        data = definition.to_dict()
        data["model_config"]["model_id"] = "tampered-model"
        with pytest.raises(ValueError, match="integrity check failed"):
            ExperimentDefinition.from_dict(data)

    def test_unsupported_schema_version_is_rejected(self):
        definition = ExperimentDefinition(
            experiment_id="exp-1",
            name="exp",
            dataset_version=DatasetVersion(dataset=_dataset()),
            model_config=_model_config(),
            evaluator_configs=_evaluator_configs(),
        )
        data = definition.to_dict()
        data["schema_version"] = 999
        with pytest.raises(ValueError, match="schema_version"):
            ExperimentDefinition.from_dict(data)


class TestExperimentRun:
    def _run(self, run_id: str = "run-1") -> ExperimentRun:
        definition = ExperimentDefinition(
            experiment_id="exp-1",
            name="exp",
            dataset_version=DatasetVersion(dataset=_dataset()),
            model_config=_model_config(),
            evaluator_configs=_evaluator_configs(),
        )
        return ExperimentRun(
            run_id=run_id,
            experiment_id=definition.experiment_id,
            status=RunStatus.SUCCEEDED,
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:01+00:00",
            dataset_version=definition.dataset_version,
            model_config=definition.model_config,
            evaluator_configs=definition.evaluator_configs,
            config_fingerprint=definition.config_fingerprint,
            result=None,
        )

    def test_requires_non_empty_run_id(self):
        with pytest.raises(ValueError, match="run_id"):
            run = self._run()
            run.run_id = ""
            run.__post_init__()

    def test_two_runs_of_the_same_experiment_have_distinct_identities(self):
        run_a = self._run(run_id="run-a")
        run_b = self._run(run_id="run-b")
        assert run_a.experiment_id == run_b.experiment_id
        assert run_a.run_id != run_b.run_id

    def test_succeeded_property_reflects_status(self):
        run = self._run()
        assert run.succeeded is True
        run.status = RunStatus.FAILED
        assert run.succeeded is False

    def test_round_trip_through_json(self):
        run = self._run()
        reloaded = ExperimentRun.from_json(run.to_json())
        assert reloaded.run_id == run.run_id
        assert reloaded.status == run.status
        assert reloaded.config_fingerprint == run.config_fingerprint

    def test_tampered_configuration_is_rejected_on_load(self):
        run = self._run()
        data = run.to_dict()
        data["evaluator_configs"] = [{"evaluator_name": "different_evaluator", "parameters": {}}]
        with pytest.raises(ValueError, match="integrity check failed"):
            ExperimentRun.from_dict(data)
