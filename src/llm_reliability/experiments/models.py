"""
Domain models for the Experiment System.

These models build a reproducible experiment on top of the existing Core
Evaluation Engine (``llm_reliability.evaluation``) without duplicating it:
``Dataset``, ``TestCase``, ``ModelConfig``, and ``RunResult`` are reused
directly. This module adds only what the evaluation engine does not need to
know about: experiment identity, dataset versioning, evaluator
configuration records, and run-level lifecycle metadata.

Conceptual model:

    ExperimentDefinition (what should be tested)
        -> DatasetVersion (content-addressed dataset)
        -> ModelConfig (reused from the evaluation engine)
        -> EvaluatorConfig(s) (stable evaluator identifiers + parameters)

    ExperimentRun (one execution of an ExperimentDefinition)
        -> a snapshot of the configuration actually used
        -> RunResult (reused from the evaluation engine)
        -> execution status

As in the evaluation engine, validation happens eagerly in
``__post_init__``, and reconstruction from persisted data goes through the
same validated constructors via ``from_dict`` rather than producing
unvalidated objects.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from llm_reliability.evaluation.models import Dataset, ModelConfig, RunResult, TestCase
from llm_reliability.experiments.fingerprint import (
    compute_dataset_fingerprint,
    compute_experiment_fingerprint,
)

SCHEMA_VERSION = 1


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _test_case_payload(test_case: TestCase) -> dict[str, Any]:
    return {
        "id": test_case.id,
        "input": test_case.input,
        "reference_answer": test_case.reference_answer,
        "metadata": test_case.metadata,
    }


@dataclass
class DatasetVersion:
    """A content-addressed, immutable version of a ``Dataset``.

    ``version_id`` is a SHA-256 fingerprint of the dataset's test cases
    (id, input, reference_answer, metadata), computed automatically and
    independent of ``dataset.name``. This is what lets a later reader
    determine exactly which dataset content an experiment run used: the
    human-readable name can change or be reused, but the version id only
    matches when the test case content is identical.
    """

    dataset: Dataset
    version_id: str = field(init=False, default="")

    def __post_init__(self) -> None:
        payload = [_test_case_payload(tc) for tc in self.dataset.test_cases]
        self.version_id = compute_dataset_fingerprint(payload)

    @property
    def dataset_name(self) -> str:
        return self.dataset.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset.name,
            "version_id": self.version_id,
            "test_cases": [asdict(tc) for tc in self.dataset.test_cases],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetVersion:
        dataset = Dataset(
            name=data["dataset_name"],
            test_cases=[TestCase.from_dict(tc) for tc in data["test_cases"]],
        )
        version = cls(dataset=dataset)
        stored_version_id = data.get("version_id")
        if stored_version_id is not None and stored_version_id != version.version_id:
            raise ValueError(
                f"Dataset version integrity check failed for '{dataset.name}': stored "
                f"version_id={stored_version_id!r} does not match recomputed "
                f"fingerprint={version.version_id!r}. The persisted record may be "
                "corrupted or was edited outside the platform."
            )
        return version


@dataclass
class EvaluatorConfig:
    """A stable, serializable record of one evaluator's identity and configuration.

    Stores only data, never the evaluator object itself: an evaluator's
    executable behavior is not part of the reproducibility record, only its
    identity (``evaluator_name``) and the parameters that configure it.
    """

    evaluator_name: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.evaluator_name or not self.evaluator_name.strip():
            raise ValueError("EvaluatorConfig.evaluator_name must be a non-empty string.")

    def to_dict(self) -> dict[str, Any]:
        return {"evaluator_name": self.evaluator_name, "parameters": self.parameters}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluatorConfig:
        return cls(
            evaluator_name=data["evaluator_name"],
            parameters=dict(data.get("parameters", {})),
        )

    @classmethod
    def from_evaluator(cls, evaluator: Any) -> EvaluatorConfig:
        """Build an ``EvaluatorConfig`` from a live ``Evaluator`` instance.

        Reads ``evaluator.name`` and ``evaluator.get_config()`` rather than
        serializing the evaluator itself.
        """
        return cls(evaluator_name=evaluator.name, parameters=evaluator.get_config())


@dataclass
class ExperimentDefinition:
    """What should be tested: a reproducible, versioned experiment definition.

    ``config_fingerprint`` is derived automatically from
    ``dataset_version``, ``model_config``, and ``evaluator_configs`` -- the
    inputs that actually determine what gets executed -- and excludes
    ``experiment_id``, ``name``, ``description``, ``metadata``, and
    ``created_at``. Two definitions with identical reproducibility-relevant
    configuration always produce the identical fingerprint, regardless of
    what they are named or when they were created.
    """

    experiment_id: str
    name: str
    dataset_version: DatasetVersion
    model_config: ModelConfig
    evaluator_configs: list[EvaluatorConfig]
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now_iso)
    config_fingerprint: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if not self.experiment_id or not self.experiment_id.strip():
            raise ValueError("ExperimentDefinition.experiment_id must be a non-empty string.")
        if not self.name or not self.name.strip():
            raise ValueError("ExperimentDefinition.name must be a non-empty string.")
        if len(self.evaluator_configs) == 0:
            raise ValueError(
                f"ExperimentDefinition '{self.experiment_id}' requires at least one "
                "evaluator configuration."
            )
        self.config_fingerprint = compute_experiment_fingerprint(
            self.dataset_version.version_id,
            asdict(self.model_config),
            [ec.to_dict() for ec in self.evaluator_configs],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "dataset_version": self.dataset_version.to_dict(),
            "model_config": asdict(self.model_config),
            "evaluator_configs": [ec.to_dict() for ec in self.evaluator_configs],
            "metadata": self.metadata,
            "created_at": self.created_at,
            "config_fingerprint": self.config_fingerprint,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentDefinition:
        schema_version = data.get("schema_version")
        if schema_version not in (None, SCHEMA_VERSION):
            raise ValueError(f"Unsupported ExperimentDefinition schema_version: {schema_version!r}")
        definition = cls(
            experiment_id=data["experiment_id"],
            name=data["name"],
            dataset_version=DatasetVersion.from_dict(data["dataset_version"]),
            model_config=ModelConfig.from_dict(data["model_config"]),
            evaluator_configs=[EvaluatorConfig.from_dict(ec) for ec in data["evaluator_configs"]],
            description=data.get("description"),
            metadata=dict(data.get("metadata", {})),
            created_at=data.get("created_at", _utc_now_iso()),
        )
        stored_fingerprint = data.get("config_fingerprint")
        if stored_fingerprint is not None and stored_fingerprint != definition.config_fingerprint:
            raise ValueError(
                f"Experiment configuration integrity check failed for "
                f"'{definition.experiment_id}': stored fingerprint={stored_fingerprint!r} "
                f"does not match recomputed fingerprint={definition.config_fingerprint!r}. "
                "The persisted record may be corrupted or was edited outside the platform."
            )
        return definition

    @classmethod
    def from_json(cls, text: str) -> ExperimentDefinition:
        return cls.from_dict(json.loads(text))


class RunStatus(StrEnum):
    """Execution status of a single ``ExperimentRun``.

    ``SUCCEEDED``: every test case executed without error.
    ``PARTIALLY_SUCCEEDED``: at least one test case succeeded and at least
    one failed; the run produced meaningful, if incomplete, results.
    ``FAILED``: no test case succeeded, or the run could not be executed at
    all (see ``ExperimentRun.error``); the run does not represent a
    meaningful evaluation of the experiment.
    """

    SUCCEEDED = "succeeded"
    PARTIALLY_SUCCEEDED = "partially_succeeded"
    FAILED = "failed"


@dataclass
class ExperimentRun:
    """One execution of an ``ExperimentDefinition``.

    Retains a full snapshot of the configuration actually used (dataset
    version, model configuration, evaluator configurations, and their
    combined fingerprint) at the time of execution, so a persisted run
    remains completely inspectable even if the originating
    ``ExperimentDefinition`` is later changed, deleted, or unavailable.

    An experiment's identity (``experiment_id``) is stable across repeated
    executions; each execution gets its own ``run_id``, so two runs of the
    same experiment are always distinguishable from each other even when
    their configuration and results are identical.
    """

    run_id: str
    experiment_id: str
    status: RunStatus
    started_at: str
    finished_at: str
    dataset_version: DatasetVersion
    model_config: ModelConfig
    evaluator_configs: list[EvaluatorConfig]
    config_fingerprint: str
    result: RunResult | None
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.run_id or not self.run_id.strip():
            raise ValueError("ExperimentRun.run_id must be a non-empty string.")
        if not self.experiment_id or not self.experiment_id.strip():
            raise ValueError("ExperimentRun.experiment_id must be a non-empty string.")

    @property
    def succeeded(self) -> bool:
        return self.status == RunStatus.SUCCEEDED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "status": self.status.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "dataset_version": self.dataset_version.to_dict(),
            "model_config": asdict(self.model_config),
            "evaluator_configs": [ec.to_dict() for ec in self.evaluator_configs],
            "config_fingerprint": self.config_fingerprint,
            "result": self.result.to_dict() if self.result is not None else None,
            "error": self.error,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentRun:
        schema_version = data.get("schema_version")
        if schema_version not in (None, SCHEMA_VERSION):
            raise ValueError(f"Unsupported ExperimentRun schema_version: {schema_version!r}")
        result_data = data.get("result")
        dataset_version = DatasetVersion.from_dict(data["dataset_version"])
        model_config = ModelConfig.from_dict(data["model_config"])
        evaluator_configs = [EvaluatorConfig.from_dict(ec) for ec in data["evaluator_configs"]]
        run = cls(
            run_id=data["run_id"],
            experiment_id=data["experiment_id"],
            status=RunStatus(data["status"]),
            started_at=data["started_at"],
            finished_at=data["finished_at"],
            dataset_version=dataset_version,
            model_config=model_config,
            evaluator_configs=evaluator_configs,
            config_fingerprint=data["config_fingerprint"],
            result=RunResult.from_dict(result_data) if result_data is not None else None,
            error=data.get("error"),
        )
        expected_fingerprint = compute_experiment_fingerprint(
            dataset_version.version_id,
            asdict(model_config),
            [ec.to_dict() for ec in evaluator_configs],
        )
        if run.config_fingerprint != expected_fingerprint:
            raise ValueError(
                f"ExperimentRun configuration integrity check failed for run "
                f"'{run.run_id}': stored fingerprint={run.config_fingerprint!r} does not "
                f"match recomputed fingerprint={expected_fingerprint!r}. The persisted "
                "record may be corrupted or was edited outside the platform."
            )
        return run

    @classmethod
    def from_json(cls, text: str) -> ExperimentRun:
        return cls.from_dict(json.loads(text))
