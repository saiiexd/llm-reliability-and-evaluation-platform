"""
Shared preparation for every API execution path (single run, benchmark, background job).

One place decides whether a run may start, so the three entry points cannot
drift apart:

1. the adapter must be allowlisted;
2. if the adapter *or any declared evaluator* (a real LLM judge) calls a model
   server, the request must opt in to external inference;
3. adapter and evaluators are rebuilt from the experiment's declared
   configuration and validated against it (``ExperimentManager.validate_execution``)
   before anything runs or is saved.
"""

from __future__ import annotations

from typing import Any

from llm_reliability.api import live
from llm_reliability.api.errors import ExecutionError
from llm_reliability.api.registry import AdapterRegistry, EvaluatorRegistry
from llm_reliability.evaluation.adapters import ModelAdapter
from llm_reliability.experiments.manager import ExperimentManager
from llm_reliability.experiments.models import ExperimentDefinition


def external_components(
    definition: ExperimentDefinition,
    adapter_id: str,
    adapter_registry: AdapterRegistry,
    eval_registry: EvaluatorRegistry,
) -> list[str]:
    """Names of the parts of this execution that would call a model server."""
    parts = [f"adapter '{adapter_id}'"] if adapter_registry.is_external_provider(adapter_id) else []
    parts += [
        f"evaluator '{ec.evaluator_name}' (judge {ec.parameters.get('judge_model_id')})"
        for ec in definition.evaluator_configs
        if eval_registry.requires_external_inference(ec)
    ]
    return parts


def prepare_execution(
    definition: ExperimentDefinition,
    adapter_id: str,
    allow_external_inference: bool,
    manager: ExperimentManager,
    adapter_registry: AdapterRegistry,
    eval_registry: EvaluatorRegistry,
) -> tuple[ModelAdapter, list[Any]]:
    """Build and validate the adapter and evaluators, or raise ``ExecutionError`` (HTTP 400)."""
    if adapter_id not in adapter_registry.list_available():
        available = ", ".join(adapter_registry.list_available())
        raise ExecutionError(f"Unknown adapter '{adapter_id}'. Allowlisted adapters: {available}")

    external = external_components(definition, adapter_id, adapter_registry, eval_registry)
    if external and not allow_external_inference:
        raise ExecutionError(
            "This execution requires external inference: it calls a model server ("
            + "; ".join(external)
            + "). Explicit opt-in is required via 'allow_external_inference=True' "
            "to permit outbound requests."
        )

    try:
        adapter = adapter_registry.get(
            adapter_id,
            config={
                "allow_external_inference": allow_external_inference,
                "model_config": definition.model_config,
            },
        )
        evaluators = [
            eval_registry.get_for_execution(ec, allow_external_inference)
            for ec in definition.evaluator_configs
        ]
        manager.validate_execution(definition, adapter, evaluators)
    except ValueError as exc:
        raise ExecutionError(f"Adapter '{adapter_id}' cannot run this experiment: {exc}") from exc
    except Exception as exc:  # e.g. the embedding server is unreachable
        raise ExecutionError(
            f"Could not prepare adapter '{adapter_id}': {type(exc).__name__}: {exc}"
        ) from exc
    return adapter, evaluators


def warm_up_local_models(
    definition: ExperimentDefinition,
    adapter_id: str,
    allow_external_inference: bool,
    eval_registry: EvaluatorRegistry,
) -> None:
    """Load every local model this execution will use, outside the recorded run.

    Only for adapters served by the configured OpenAI-compatible server and
    for live judges (which always use it); never for hosted APIs, where a
    warm-up request would cost money and model loading is not visible.
    """
    generators: list[str] = []
    embeddings: list[str] = []
    if adapter_id in live.LOCAL_ADAPTERS:
        generators.append(definition.model_config.model_id)
        rag = definition.model_config.extra_params.get("rag")
        if adapter_id == "openai-custom-rag" and isinstance(rag, dict):
            embeddings.append(str(rag.get("embedding_model_id")))
    generators += [
        str(ec.parameters.get("judge_model_id"))
        for ec in definition.evaluator_configs
        if eval_registry.requires_external_inference(ec)
    ]
    live.warm_up(generators, embeddings, allow_external_inference)
