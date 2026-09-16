"""Tests for LocalFileExperimentRepository."""

import json

import pytest

from llm_reliability.evaluation import Dataset, ModelConfig, TestCase
from llm_reliability.experiments.local_repository import LocalFileExperimentRepository
from llm_reliability.experiments.models import (
    DatasetVersion,
    EvaluatorConfig,
    ExperimentDefinition,
    ExperimentRun,
    RunStatus,
)
from llm_reliability.experiments.repository import (
    CorruptedRecordError,
    ExperimentAlreadyExistsError,
    ExperimentNotFoundError,
    RunAlreadyExistsError,
    RunNotFoundError,
)


def _definition(experiment_id: str = "exp-1") -> ExperimentDefinition:
    dataset = Dataset(name="ds", test_cases=[TestCase(id="t1", input="q", reference_answer="a")])
    return ExperimentDefinition(
        experiment_id=experiment_id,
        name="Test experiment",
        dataset_version=DatasetVersion(dataset=dataset),
        model_config=ModelConfig(model_id="mock-adapter-v1"),
        evaluator_configs=[EvaluatorConfig(evaluator_name="exact_match")],
    )


def _run(definition: ExperimentDefinition, run_id: str = "run-1") -> ExperimentRun:
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


class TestExperimentPersistence:
    def test_save_and_load_round_trip(self, tmp_path):
        repo = LocalFileExperimentRepository(tmp_path)
        definition = _definition()
        repo.save_experiment(definition)

        loaded = repo.load_experiment("exp-1")
        assert loaded.experiment_id == definition.experiment_id
        assert loaded.config_fingerprint == definition.config_fingerprint

    def test_load_missing_experiment_raises_not_found(self, tmp_path):
        repo = LocalFileExperimentRepository(tmp_path)
        with pytest.raises(ExperimentNotFoundError):
            repo.load_experiment("does-not-exist")

    def test_saving_duplicate_experiment_id_is_rejected(self, tmp_path):
        repo = LocalFileExperimentRepository(tmp_path)
        repo.save_experiment(_definition())
        with pytest.raises(ExperimentAlreadyExistsError):
            repo.save_experiment(_definition())

    def test_persisted_file_is_valid_readable_json(self, tmp_path):
        repo = LocalFileExperimentRepository(tmp_path)
        repo.save_experiment(_definition())
        path = tmp_path / "experiments" / "exp-1.json"
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["experiment_id"] == "exp-1"

    def test_corrupted_json_raises_corrupted_record_error(self, tmp_path):
        repo = LocalFileExperimentRepository(tmp_path)
        repo.save_experiment(_definition())
        path = tmp_path / "experiments" / "exp-1.json"
        path.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(CorruptedRecordError):
            repo.load_experiment("exp-1")

    def test_structurally_invalid_json_raises_corrupted_record_error(self, tmp_path):
        repo = LocalFileExperimentRepository(tmp_path)
        repo.save_experiment(_definition())
        path = tmp_path / "experiments" / "exp-1.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        del data["model_config"]
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(CorruptedRecordError):
            repo.load_experiment("exp-1")

    def test_tampered_content_raises_corrupted_record_error(self, tmp_path):
        repo = LocalFileExperimentRepository(tmp_path)
        repo.save_experiment(_definition())
        path = tmp_path / "experiments" / "exp-1.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["model_config"]["model_id"] = "tampered"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(CorruptedRecordError, match="integrity check failed"):
            repo.load_experiment("exp-1")

    def test_list_experiment_ids(self, tmp_path):
        repo = LocalFileExperimentRepository(tmp_path)
        repo.save_experiment(_definition("exp-a"))
        repo.save_experiment(_definition("exp-b"))
        assert repo.list_experiment_ids() == ["exp-a", "exp-b"]

    def test_list_experiment_ids_empty_store(self, tmp_path):
        repo = LocalFileExperimentRepository(tmp_path)
        assert repo.list_experiment_ids() == []

    def test_path_traversal_identifier_is_rejected(self, tmp_path):
        repo = LocalFileExperimentRepository(tmp_path)
        with pytest.raises(ValueError):
            repo.load_experiment("../outside")


class TestRunPersistence:
    def test_save_and_load_round_trip(self, tmp_path):
        repo = LocalFileExperimentRepository(tmp_path)
        definition = _definition()
        repo.save_experiment(definition)
        run = _run(definition)
        repo.save_run(run)

        loaded = repo.load_run("run-1")
        assert loaded.run_id == "run-1"
        assert loaded.status == RunStatus.SUCCEEDED

    def test_load_missing_run_raises_not_found(self, tmp_path):
        repo = LocalFileExperimentRepository(tmp_path)
        with pytest.raises(RunNotFoundError):
            repo.load_run("does-not-exist")

    def test_saving_duplicate_run_id_is_rejected(self, tmp_path):
        repo = LocalFileExperimentRepository(tmp_path)
        definition = _definition()
        repo.save_experiment(definition)
        repo.save_run(_run(definition))
        with pytest.raises(RunAlreadyExistsError):
            repo.save_run(_run(definition))

    def test_corrupted_run_raises_corrupted_record_error(self, tmp_path):
        repo = LocalFileExperimentRepository(tmp_path)
        definition = _definition()
        repo.save_experiment(definition)
        repo.save_run(_run(definition))
        path = tmp_path / "runs" / "exp-1" / "run-1.json"
        path.write_text("not json at all", encoding="utf-8")
        with pytest.raises(CorruptedRecordError):
            repo.load_run("run-1")

    def test_list_run_ids_filtered_by_experiment(self, tmp_path):
        repo = LocalFileExperimentRepository(tmp_path)
        definition_a = _definition("exp-a")
        definition_b = _definition("exp-b")
        repo.save_experiment(definition_a)
        repo.save_experiment(definition_b)
        repo.save_run(_run(definition_a, "run-a1"))
        repo.save_run(_run(definition_b, "run-b1"))

        assert repo.list_run_ids("exp-a") == ["run-a1"]
        assert repo.list_run_ids("exp-b") == ["run-b1"]
        assert sorted(repo.list_run_ids()) == ["run-a1", "run-b1"]

    def test_list_run_ids_empty_store(self, tmp_path):
        repo = LocalFileExperimentRepository(tmp_path)
        assert repo.list_run_ids() == []
        assert repo.list_run_ids("no-such-experiment") == []
