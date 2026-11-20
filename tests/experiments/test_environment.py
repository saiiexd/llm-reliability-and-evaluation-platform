"""Tests for execution-environment provenance on runs."""

import json

from llm_reliability.evaluation import (
    Dataset,
    ExactMatchEvaluator,
    MockAdapter,
    ModelConfig,
    TestCase,
)
from llm_reliability.experiments.environment import capture_environment
from llm_reliability.experiments.local_repository import LocalFileExperimentRepository
from llm_reliability.experiments.manager import ExperimentManager
from llm_reliability.experiments.models import ExperimentRun


def _definition(manager: ExperimentManager):
    return manager.create_experiment(
        experiment_id="exp-env",
        name="env",
        dataset=Dataset(name="d", test_cases=[TestCase(id="t1", input="q", reference_answer="a")]),
        model_config=ModelConfig(model_id="mock-adapter-v1"),
        evaluators=[ExactMatchEvaluator()],
    )


def test_capture_environment_is_json_serializable_and_has_core_fields():
    env = capture_environment()
    json.dumps(env)
    assert env["python"].count(".") == 2
    assert env["platform_version"]  # the package is installed in the test environment
    assert "pydantic" in env["dependencies"]


def test_capture_environment_records_no_secrets_or_env_vars(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-never-appear")
    assert "sk-should-never-appear" not in json.dumps(capture_environment())


def test_execution_records_environment_and_it_round_trips(tmp_path):
    repo = LocalFileExperimentRepository(tmp_path)
    manager = ExperimentManager(repo)
    run = manager.execute_experiment(
        _definition(manager), MockAdapter(responses={"t1": "a"}), [ExactMatchEvaluator()]
    )
    assert run.environment is not None
    assert run.environment["python"]
    reloaded = repo.load_run(run.run_id)
    assert reloaded.environment == run.environment


def test_environment_is_not_part_of_the_fingerprint(tmp_path):
    manager = ExperimentManager(LocalFileExperimentRepository(tmp_path))
    definition = _definition(manager)
    a = manager.execute_experiment(
        definition,
        MockAdapter(responses={"t1": "a"}),
        [ExactMatchEvaluator()],
        environment={"python": "3.13.0"},
    )
    b = manager.execute_experiment(
        definition,
        MockAdapter(responses={"t1": "a"}),
        [ExactMatchEvaluator()],
        environment={"python": "3.14.0"},
    )
    assert a.config_fingerprint == b.config_fingerprint


def test_runs_saved_before_the_field_existed_still_load(tmp_path):
    repo = LocalFileExperimentRepository(tmp_path)
    manager = ExperimentManager(repo)
    run = manager.execute_experiment(
        _definition(manager), MockAdapter(responses={"t1": "a"}), [ExactMatchEvaluator()]
    )
    data = run.to_dict()
    del data["environment"]
    assert ExperimentRun.from_dict(data).environment is None
