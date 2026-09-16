"""
Experiment lifecycle orchestration.

``ExperimentManager`` is responsible for experiment-level concerns: creating
and validating experiment definitions, turning a definition into an
``ExperimentRun`` by delegating execution to the existing
``EvaluationRunner``, and passing both to a persistence layer. It contains
no evaluation logic of its own -- test case execution and evaluator
application remain entirely the responsibility of ``EvaluationRunner``; this
module only interprets the ``RunResult`` it produces to determine run
status and packages it into a reproducibility record.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from llm_reliability.evaluation.adapters import ModelAdapter
from llm_reliability.evaluation.evaluators import Evaluator
from llm_reliability.evaluation.models import Dataset, ModelConfig, RunResult
from llm_reliability.evaluation.runner import EvaluationRunner
from llm_reliability.experiments.models import (
    DatasetVersion,
    EvaluatorConfig,
    ExperimentDefinition,
    ExperimentRun,
    RunStatus,
)
from llm_reliability.experiments.repository import ExperimentRepository


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _determine_status(result: RunResult | None) -> RunStatus:
    if result is None or result.num_succeeded == 0:
        return RunStatus.FAILED
    if result.num_failed == 0:
        return RunStatus.SUCCEEDED
    return RunStatus.PARTIALLY_SUCCEEDED


class ExperimentManager:
    """Creates, executes, and persists experiments using a given repository."""

    def __init__(self, repository: ExperimentRepository) -> None:
        self._repository = repository

    def create_experiment(
        self,
        experiment_id: str,
        name: str,
        dataset: Dataset,
        model_config: ModelConfig,
        evaluators: Sequence[Evaluator],
        description: str | None = None,
        metadata: dict | None = None,
    ) -> ExperimentDefinition:
        """Define and persist a new experiment.

        ``evaluators`` are the live evaluator instances the experiment is
        defined against; their ``name`` and ``get_config()`` are recorded as
        the experiment's evaluator configuration, but the instances
        themselves are not retained -- ``execute_experiment`` requires them
        to be supplied again at execution time.

        Raises ``ExperimentAlreadyExistsError`` (from the repository) if
        ``experiment_id`` is already in use.
        """
        definition = ExperimentDefinition(
            experiment_id=experiment_id,
            name=name,
            dataset_version=DatasetVersion(dataset=dataset),
            model_config=model_config,
            evaluator_configs=[EvaluatorConfig.from_evaluator(e) for e in evaluators],
            description=description,
            metadata=dict(metadata) if metadata else {},
        )
        self._repository.save_experiment(definition)
        return definition

    def execute_experiment(
        self,
        definition: ExperimentDefinition,
        adapter: ModelAdapter,
        evaluators: Sequence[Evaluator],
        run_id: str | None = None,
    ) -> ExperimentRun:
        """Execute ``definition`` once via ``EvaluationRunner`` and persist the run.

        ``evaluators`` must match ``definition.evaluator_configs`` by name
        and order; this is a consistency check, not a re-derivation of
        configuration, so that a run can never be silently recorded against
        evaluators different from the ones the experiment declares. Raises
        ``ValueError`` if they do not match.

        ``run_id`` defaults to a freshly generated, random identifier: run
        identity only needs to be unique, not reproducible, since two runs
        of the same experiment are expected to have different run ids even
        when their configuration is identical.
        """
        configured_names = [ec.evaluator_name for ec in definition.evaluator_configs]
        supplied_names = [e.name for e in evaluators]
        if configured_names != supplied_names:
            raise ValueError(
                f"Evaluator mismatch for experiment '{definition.experiment_id}': "
                f"definition specifies {configured_names}, but {supplied_names} were "
                "supplied for execution."
            )

        if run_id is None:
            run_id = uuid.uuid4().hex

        started_at = _utc_now_iso()
        runner = EvaluationRunner(adapter=adapter, evaluators=list(evaluators))
        result: RunResult | None
        error: str | None
        try:
            result = runner.run(definition.dataset_version.dataset, definition.model_config)
            error = None
        except Exception as exc:
            result = None
            error = f"{type(exc).__name__}: {exc}"
        finished_at = _utc_now_iso()

        run = ExperimentRun(
            run_id=run_id,
            experiment_id=definition.experiment_id,
            status=_determine_status(result),
            started_at=started_at,
            finished_at=finished_at,
            dataset_version=definition.dataset_version,
            model_config=definition.model_config,
            evaluator_configs=definition.evaluator_configs,
            config_fingerprint=definition.config_fingerprint,
            result=result,
            error=error,
        )
        self._repository.save_run(run)
        return run

    def load_experiment(self, experiment_id: str) -> ExperimentDefinition:
        return self._repository.load_experiment(experiment_id)

    def load_run(self, run_id: str) -> ExperimentRun:
        return self._repository.load_run(run_id)
