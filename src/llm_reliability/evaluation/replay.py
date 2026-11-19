"""
Replay: re-evaluate the answers of a persisted run without generating again.

Evaluator studies (RQ3) need the *same* answers scored by different
evaluators or evaluator versions. Regenerating would mix generation noise into
an evaluator comparison (temperature 0 is not fully deterministic, see
EXP-001), so ``ReplayAdapter`` returns, for each test case, the exact
``ModelResponse`` a source run recorded.

Provenance is explicit: every replayed response keeps the source's model id,
token counts and retrieval evidence, and adds
``provider_metadata["replay"] = {"source_run_id", "source_experiment_id"}``.
``latency_ms`` is the *original* generation latency, not a new measurement;
operational analyses must therefore exclude replay runs (they are identifiable
by that metadata and by their adapter id).

The adapter also exposes the source's RAG configuration (``config``) or
closed-book template (``template_id``) so that ``ExperimentManager`` can verify
that the replay experiment declares the same generation setup as the source.
A test case whose source generation failed is replayed as a failure, never as
an answer.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from llm_reliability.evaluation.adapters import AdapterError, ModelAdapter
from llm_reliability.evaluation.models import ExecutionRequest, ModelResponse


class ReplayAdapter(ModelAdapter):
    """Serves the recorded responses of one persisted ``ExperimentRun``."""

    def __init__(self, source_run: Any) -> None:
        if source_run.result is None:
            raise ValueError(f"Run '{source_run.run_id}' has no results to replay.")
        self._source_run_id: str = source_run.run_id
        self._source_experiment_id: str = source_run.experiment_id
        self._responses: dict[str, ModelResponse | None] = {
            tc.test_case_id: tc.model_response for tc in source_run.result.test_case_results
        }
        extra = source_run.model_config.extra_params
        rag = extra.get("rag")
        if isinstance(rag, dict):
            from llm_reliability.rag.pipeline import RagPipelineConfig

            self.config = RagPipelineConfig.from_dict(rag)
        template = extra.get("prompt_template_id")
        if template is not None:
            self.template_id = str(template)

    @property
    def source_run_id(self) -> str:
        return self._source_run_id

    def generate(self, request: ExecutionRequest) -> ModelResponse:
        if request.test_case_id not in self._responses:
            raise AdapterError(
                f"Test case '{request.test_case_id}' is not in source run '{self._source_run_id}'."
            )
        recorded = self._responses[request.test_case_id]
        if recorded is None:
            raise AdapterError(
                f"Generation failed for '{request.test_case_id}' in source run "
                f"'{self._source_run_id}'; there is no answer to replay."
            )
        metadata = dict(recorded.provider_metadata)
        metadata["replay"] = {
            "source_run_id": self._source_run_id,
            "source_experiment_id": self._source_experiment_id,
        }
        return replace(recorded, provider_metadata=metadata)
