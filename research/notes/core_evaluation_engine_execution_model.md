# Core Evaluation Engine: Execution Model

## Purpose of This Note

This note documents the conceptual execution model implemented by
`src/llm_reliability/evaluation/`, and draws an explicit line between two things that are
easy to conflate: **execution infrastructure** (how a dataset gets run against a model and
produces structured results) and **evaluation methodology** (how good those results
actually are, scientifically). This milestone builds only the former. The latter is a
separate, much larger body of work that later milestones will address.

## The Execution Pipeline

```
Dataset -> TestCase -> ExecutionRequest -> ModelAdapter -> ModelResponse
    -> Evaluator(s) -> EvaluationResult -> TestCaseResult -> RunResult
```

- A **Dataset** is an ordered, validated collection of **TestCase** records. Each test
  case has a stable id, an input, and an optional reference answer. Ids must be unique
  within a dataset so every result can always be traced back to the test case that
  produced it, even after failures, filtering, or later persistence.
- A **TestCase** is deliberately narrowed to an **ExecutionRequest** (id, input text, and
  model configuration) before being handed to a **ModelAdapter**. This narrowing is
  intentional: an adapter that generates answers must not depend on fields, such as a
  reference answer, that are only meaningful to evaluation.
- A **ModelAdapter** is the only component that is allowed to depend on a specific model
  or application. The engine itself depends only on the adapter interface. This is what
  makes the engine provider-agnostic: swapping a real LLM provider in for the current
  `MockAdapter` requires no change to `EvaluationRunner`, `Evaluator`, or any domain model.
- A **ModelResponse** captures what the adapter actually produced, plus whatever metadata
  the provider actually reported (latency, token counts, provider-specific detail). Fields
  a provider does not report are represented as `None`, never fabricated. This matters for
  reproducibility: a downstream analysis must be able to tell "not measured" apart from
  "measured as zero."
- One or more **Evaluators** each independently inspect a `TestCase` and its
  `ModelResponse` and produce an **EvaluationResult**. Evaluators do not depend on each
  other or on adapter internals, so arbitrary evaluators can be composed over the same
  response.
- **EvaluationRunner** orchestrates the above for an entire dataset, and isolates failure:
  if generation or evaluation fails for one test case, that failure is recorded explicitly
  on that test case's result, and every other test case in the run proceeds normally. A
  run is never silently corrupted or aborted by a single bad input or provider error.
- The **RunResult** is the complete, structured record of a run: every test case's input,
  reference (if any), generated response (if any), every evaluator's result, and any
  execution error. It supports deterministic JSON serialization so a run can later be
  written to disk, diffed, or loaded into a persistence layer without ambiguity.

## Execution Infrastructure vs. Evaluation Methodology

The distinction matters because the project's central principle (see
`research/project_definition.md`) is that trust in LLM systems, and in the metrics used to
assess them, must be earned through measurement rather than assumed. That principle
applies as much to this platform's own evaluators as to the models it tests.

- **Execution infrastructure** answers: "did we run this test case correctly, capture
  what actually happened, and hand it to an evaluator without corruption or ambiguity?"
  This is what `EvaluationRunner`, the adapters, and the domain models in this milestone
  provide. It is a solved, mechanical problem once designed correctly, and it does not
  itself make any claim about model quality.
- **Evaluation methodology** answers: "does a given evaluator's judgment actually
  correspond to something we care about (correctness, faithfulness, absence of
  hallucination), and how do we know?" This is a research problem, not an engineering one.
  `ExactMatchEvaluator`, included in this milestone, is a deliberately minimal baseline
  that exists only to exercise the execution infrastructure end to end. It measures
  literal string identity, which is a poor proxy for answer quality: it cannot recognize
  paraphrase, partial correctness, or semantic equivalence, and it is not applicable at
  all when no reference answer exists. It must not be cited as evidence about how good a
  model's answers are.

Later milestones will introduce evaluators with real methodological weight (for example,
LLM-as-a-judge, faithfulness scoring against retrieved context, or semantic similarity),
and each of those will need its own validation against human judgment before its scores
can be trusted, per the project's evaluation-methods-must-be-evaluated principle. That
validation work is out of scope here. This milestone's job was only to make sure that once
such an evaluator exists, it can be plugged into `EvaluationRunner` and produce a
`RunResult` without any change to the surrounding infrastructure.

## Deliberately Excluded From This Milestone

To keep the execution model legible and avoid speculative architecture, this milestone
does not implement:

- Retrieval-augmented generation execution or retrieved-context handling (a future RAG
  adapter can carry retrieval information in `ModelResponse.provider_metadata` without any
  change to this schema).
- Persistence of results beyond in-memory objects and their JSON serialization.
- Reliability, consistency, or statistical analysis across runs.
- Observability or tracing of adapter/provider internals.
- Any evaluator beyond the exact-match baseline.
- A real LLM provider adapter (only the deterministic `MockAdapter` exists so far).

These are not omissions to be apologized for; they are the intended scope boundary of this
milestone, chosen so the execution model could be validated on its own before any of the
above is built on top of it.
