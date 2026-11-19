"""
Prompt templating for non-RAG (closed-book) execution.

``RagPipeline`` builds its prompt from retrieved context. A closed-book
baseline -- the same questions asked without any retrieval -- needs the same
kind of explicit, versioned prompt so the two arms differ only in whether
context is supplied, not in answer-format instructions. Without such a
baseline the effect of retrieval cannot be separated from what the model
already knows (and public QA sets are likely to have been seen in training).

``PromptTemplateAdapter`` wraps any ``ModelAdapter``: it renders the
template around the test case input, calls the wrapped adapter once, and
records the template id and exact prompt in ``provider_metadata`` so the
prompt is auditable after the fact.
"""

from __future__ import annotations

from llm_reliability.evaluation.adapters import ModelAdapter
from llm_reliability.evaluation.models import ExecutionRequest, ModelResponse

CLOSED_BOOK_TEMPLATES: dict[str, str] = {
    "closed_book_concise_v1": (
        "Answer the question. Reply with only the answer, as a short phrase, with no "
        "explanation. If you do not know, reply exactly: I don't know\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
}


def render_closed_book_prompt(template_id: str, question: str) -> str:
    if template_id not in CLOSED_BOOK_TEMPLATES:
        raise ValueError(f"Unknown closed-book prompt template '{template_id}'.")
    return CLOSED_BOOK_TEMPLATES[template_id].format(question=question)


class PromptTemplateAdapter(ModelAdapter):
    """Wraps a generation adapter with an explicit, versioned prompt template."""

    def __init__(self, generation_adapter: ModelAdapter, template_id: str) -> None:
        render_closed_book_prompt(template_id, "validation")  # fail fast on unknown ids
        self._generation_adapter = generation_adapter
        self._template_id = template_id

    @property
    def template_id(self) -> str:
        return self._template_id

    def generate(self, request: ExecutionRequest) -> ModelResponse:
        prompt = render_closed_book_prompt(self._template_id, request.input_text)
        response = self._generation_adapter.generate(
            ExecutionRequest(
                test_case_id=request.test_case_id,
                input_text=prompt,
                model_config=request.model_config,
            )
        )
        metadata = dict(response.provider_metadata)
        metadata["prompt_template_id"] = self._template_id
        metadata["prompt"] = prompt
        return ModelResponse(
            test_case_id=response.test_case_id,
            output_text=response.output_text,
            model_id=response.model_id,
            latency_ms=response.latency_ms,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            provider_metadata=metadata,
        )
