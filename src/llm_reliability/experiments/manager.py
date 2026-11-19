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
from typing import Any

from llm_reliability.evaluation.adapters import ModelAdapter
from llm_reliability.evaluation.evaluators import Evaluator
from llm_reliability.evaluation.models import Dataset, ModelConfig, RunResult
from llm_reliability.evaluation.runner import EvaluationRunner
from llm_reliability.experiments.environment import capture_environment
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


# Evaluator config keys added after experiments had already been persisted
# (judge_model_config: 2026-11-19). Absent from an old record = not compared.
_LATER_ADDED_CONFIG_KEYS = frozenset({"judge_model_config"})


def _check_rag_config_matches(definition: ExperimentDefinition, adapter: ModelAdapter) -> None:
    """Refuse to run a RAG pipeline whose retrieval settings differ from the definition's.

    A RAG experiment records its pipeline configuration (top_k, chunking,
    embedding model, corpus fingerprint, prompt template) in
    ``model_config.extra_params["rag"]``, and that record is part of the
    configuration fingerprint. Executing it with a pipeline configured
    differently (for example top_k=1 for a top_k=2 experiment) would store a
    run whose recorded configuration is not the one that produced it.

    Duck-typed (``adapter.config.to_dict()`` with a ``corpus_fingerprint``) to
    avoid an import cycle with ``llm_reliability.rag``. Non-RAG adapters, and
    definitions that declare no RAG configuration, are not checked.
    """
    declared = definition.model_config.extra_params.get("rag")
    adapter_config = getattr(adapter, "config", None)
    is_rag_adapter = hasattr(adapter_config, "corpus_fingerprint")
    if isinstance(declared, dict) and not is_rag_adapter:
        raise ValueError(
            f"Experiment '{definition.experiment_id}' declares a RAG configuration, but the "
            "supplied adapter performs no retrieval. Its answers would be recorded as RAG "
            "answers without any retrieved context. Choose a RAG adapter."
        )
    if not isinstance(declared, dict) or not is_rag_adapter:
        return
    actual = adapter_config.to_dict()  # type: ignore[union-attr]
    differing = sorted(k for k in set(declared) | set(actual) if declared.get(k) != actual.get(k))
    if differing:
        details = ", ".join(
            f"{k}: declared {declared.get(k)!r}, pipeline {actual.get(k)!r}" for k in differing
        )
        raise ValueError(
            f"RAG pipeline configuration does not match experiment "
            f"'{definition.experiment_id}' ({details}). Choose the adapter that matches the "
            "experiment, or define a new experiment for this configuration."
        )


def _check_prompt_template_matches(definition: ExperimentDefinition, adapter: ModelAdapter) -> None:
    """Refuse to run a closed-book experiment without the prompt template it declares.

    ``model_config.extra_params["prompt_template_id"]`` is part of the
    fingerprint; the adapter must actually apply it (``PromptTemplateAdapter``
    exposes ``template_id``). RAG pipelines carry their template in the RAG
    configuration instead, which ``_check_rag_config_matches`` verifies.
    """
    declared = definition.model_config.extra_params.get("prompt_template_id")
    if declared is None or hasattr(getattr(adapter, "config", None), "corpus_fingerprint"):
        return
    actual = getattr(adapter, "template_id", None)
    if actual != declared:
        raise ValueError(
            f"Experiment '{definition.experiment_id}' declares prompt template {declared!r}, but "
            f"the supplied adapter applies {actual!r}. Choose an adapter that applies the "
            "declared template."
        )


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

    def validate_execution(
        self,
        definition: ExperimentDefinition,
        adapter: ModelAdapter,
        evaluators: Sequence[Evaluator],
    ) -> None:
        """Raise ``ValueError`` if executing ``definition`` with these components would
        record a run under a configuration that did not produce it.

        Called by ``execute_experiment``; callers that execute several adapters
        (benchmarks) should call it for every adapter *before* executing any, so a
        mismatch does not leave a partially persisted benchmark behind.
        """
        configured_names = [ec.evaluator_name for ec in definition.evaluator_configs]
        supplied_names = [e.name for e in evaluators]
        if configured_names != supplied_names:
            raise ValueError(
                f"Evaluator mismatch for experiment '{definition.experiment_id}': "
                f"definition specifies {configured_names}, but {supplied_names} were "
                "supplied for execution."
            )

        # Names alone are not enough: the same evaluator name can be configured
        # differently (e.g. a canned fake judge vs a real judge model, or a
        # different threshold). Executing with a different configuration than
        # the definition declares would record the run against settings that
        # did not produce it, so it is refused.
        for declared, supplied in zip(definition.evaluator_configs, evaluators, strict=True):
            supplied_config = EvaluatorConfig.from_evaluator(supplied).parameters
            # Records written before a config key was introduced lack it; drop
            # only those known later-added keys before comparing, so older
            # experiments stay executable while every key they DO declare must
            # still match exactly.
            supplied_config = {
                k: v
                for k, v in supplied_config.items()
                if k in declared.parameters or k not in _LATER_ADDED_CONFIG_KEYS
            }
            if supplied_config != declared.parameters:
                raise ValueError(
                    f"Evaluator configuration mismatch for '{declared.evaluator_name}' in "
                    f"experiment '{definition.experiment_id}': the experiment declares "
                    f"{declared.parameters!r} but the supplied evaluator is configured as "
                    f"{supplied_config!r}. Define a new experiment for the new configuration."
                )
        _check_rag_config_matches(definition, adapter)
        _check_prompt_template_matches(definition, adapter)

    def execute_experiment(
        self,
        definition: ExperimentDefinition,
        adapter: ModelAdapter,
        evaluators: Sequence[Evaluator],
        run_id: str | None = None,
        benchmark_id: str | None = None,
        repetition_index: int = 0,
        adapter_id: str | None = None,
        environment: dict[str, Any] | None = None,
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
        self.validate_execution(definition, adapter, evaluators)
        if environment is None:
            environment = capture_environment()
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
            benchmark_id=benchmark_id,
            repetition_index=repetition_index,
            adapter_id=adapter_id,
            environment=environment,
        )
        self._repository.save_run(run)
        return run

    def load_experiment(self, experiment_id: str) -> ExperimentDefinition:
        return self._repository.load_experiment(experiment_id)

    def load_run(self, run_id: str) -> ExperimentRun:
        return self._repository.load_run(run_id)
