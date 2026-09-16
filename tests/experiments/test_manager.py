"""Tests for ExperimentManager orchestration."""

import pytest

from llm_reliability.evaluation import (
    Dataset,
    ExactMatchEvaluator,
    MockAdapter,
    ModelConfig,
    TestCase,
)
from llm_reliability.experiments.local_repository import LocalFileExperimentRepository
from llm_reliability.experiments.manager import ExperimentManager
from llm_reliability.experiments.models import RunStatus


def _dataset(*ids: str) -> Dataset:
    return Dataset(
        name="manager-tests",
        test_cases=[TestCase(id=i, input=f"input-{i}", reference_answer=i) for i in ids],
    )


@pytest.fixture
def manager(tmp_path) -> ExperimentManager:
    return ExperimentManager(LocalFileExperimentRepository(tmp_path))


class TestCreateExperiment:
    def test_create_persists_and_returns_definition(self, manager):
        definition = manager.create_experiment(
            experiment_id="exp-1",
            name="Test experiment",
            dataset=_dataset("t1"),
            model_config=ModelConfig(model_id="mock-adapter-v1"),
            evaluators=[ExactMatchEvaluator()],
        )
        assert definition.experiment_id == "exp-1"
        loaded = manager.load_experiment("exp-1")
        assert loaded.config_fingerprint == definition.config_fingerprint

    def test_duplicate_experiment_id_is_rejected(self, manager):
        manager.create_experiment(
            experiment_id="exp-1",
            name="First",
            dataset=_dataset("t1"),
            model_config=ModelConfig(model_id="mock-adapter-v1"),
            evaluators=[ExactMatchEvaluator()],
        )
        with pytest.raises(Exception, match="already exists"):
            manager.create_experiment(
                experiment_id="exp-1",
                name="Second, different definition",
                dataset=_dataset("t2"),
                model_config=ModelConfig(model_id="mock-adapter-v1"),
                evaluators=[ExactMatchEvaluator()],
            )


class TestExecuteExperiment:
    def test_successful_execution(self, manager):
        definition = manager.create_experiment(
            experiment_id="exp-1",
            name="exp",
            dataset=_dataset("t1"),
            model_config=ModelConfig(model_id="mock-adapter-v1"),
            evaluators=[ExactMatchEvaluator()],
        )
        adapter = MockAdapter(responses={"t1": "t1"})
        run = manager.execute_experiment(definition, adapter, [ExactMatchEvaluator()])

        assert run.status == RunStatus.SUCCEEDED
        assert run.experiment_id == "exp-1"
        assert run.result is not None
        assert run.result.num_succeeded == 1
        assert run.error is None
        assert run.started_at <= run.finished_at

    def test_partial_execution_with_some_failures(self, manager):
        definition = manager.create_experiment(
            experiment_id="exp-1",
            name="exp",
            dataset=_dataset("t1", "t2"),
            model_config=ModelConfig(model_id="mock-adapter-v1"),
            evaluators=[ExactMatchEvaluator()],
        )
        adapter = MockAdapter(responses={"t1": "t1"}, errors={"t2": "boom"})
        run = manager.execute_experiment(definition, adapter, [ExactMatchEvaluator()])

        assert run.status == RunStatus.PARTIALLY_SUCCEEDED
        assert run.result.num_succeeded == 1
        assert run.result.num_failed == 1

    def test_fully_failed_execution(self, manager):
        definition = manager.create_experiment(
            experiment_id="exp-1",
            name="exp",
            dataset=_dataset("t1", "t2"),
            model_config=ModelConfig(model_id="mock-adapter-v1"),
            evaluators=[ExactMatchEvaluator()],
        )
        adapter = MockAdapter(errors={"t1": "boom", "t2": "boom"})
        run = manager.execute_experiment(definition, adapter, [ExactMatchEvaluator()])

        assert run.status == RunStatus.FAILED
        assert run.result.num_succeeded == 0

    def test_evaluator_mismatch_is_rejected(self, manager):
        class OtherEvaluator(ExactMatchEvaluator):
            name = "other_evaluator"

        definition = manager.create_experiment(
            experiment_id="exp-1",
            name="exp",
            dataset=_dataset("t1"),
            model_config=ModelConfig(model_id="mock-adapter-v1"),
            evaluators=[ExactMatchEvaluator()],
        )
        adapter = MockAdapter(responses={"t1": "t1"})
        with pytest.raises(ValueError, match="Evaluator mismatch"):
            manager.execute_experiment(definition, adapter, [OtherEvaluator()])

    def test_two_runs_of_same_experiment_have_distinct_run_ids(self, manager):
        definition = manager.create_experiment(
            experiment_id="exp-1",
            name="exp",
            dataset=_dataset("t1"),
            model_config=ModelConfig(model_id="mock-adapter-v1"),
            evaluators=[ExactMatchEvaluator()],
        )
        adapter = MockAdapter(responses={"t1": "t1"})
        run_a = manager.execute_experiment(definition, adapter, [ExactMatchEvaluator()])
        run_b = manager.execute_experiment(definition, adapter, [ExactMatchEvaluator()])

        assert run_a.run_id != run_b.run_id
        assert run_a.experiment_id == run_b.experiment_id == "exp-1"
        assert run_a.config_fingerprint == run_b.config_fingerprint

    def test_explicit_run_id_is_honored(self, manager):
        definition = manager.create_experiment(
            experiment_id="exp-1",
            name="exp",
            dataset=_dataset("t1"),
            model_config=ModelConfig(model_id="mock-adapter-v1"),
            evaluators=[ExactMatchEvaluator()],
        )
        adapter = MockAdapter(responses={"t1": "t1"})
        run = manager.execute_experiment(
            definition, adapter, [ExactMatchEvaluator()], run_id="explicit-run-id"
        )
        assert run.run_id == "explicit-run-id"
        assert manager.load_run("explicit-run-id").run_id == "explicit-run-id"

    def test_run_is_persisted_and_reloadable(self, manager):
        definition = manager.create_experiment(
            experiment_id="exp-1",
            name="exp",
            dataset=_dataset("t1"),
            model_config=ModelConfig(model_id="mock-adapter-v1"),
            evaluators=[ExactMatchEvaluator()],
        )
        adapter = MockAdapter(responses={"t1": "t1"})
        run = manager.execute_experiment(definition, adapter, [ExactMatchEvaluator()])
        reloaded = manager.load_run(run.run_id)
        assert reloaded.status == run.status
        assert reloaded.result.num_succeeded == run.result.num_succeeded
