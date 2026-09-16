# RAG Execution and Retrieval Evaluation Model

## Purpose of This Note

This note explains the conceptual model implemented by
`src/llm_reliability/rag/`: how retrieval and generation are kept as
separate, independently observable stages; how retrieval quality is
evaluated independently from answer quality; what Hit@K, Recall@K, and
Mean Reciprocal Rank (MRR) precisely measure; and how the failure
diagnosis layer distinguishes retrieval failure from generation failure
using only the evidence actually available -- including when it must
honestly report that it cannot tell.

## The Execution Model

```
Question -> Retrieval -> Retrieved Context -> Generation -> Answer
```

`RagPipeline` (`llm_reliability.rag.pipeline`) implements exactly this,
as two separate calls: it asks a `Retriever` to rank a fixed chunk corpus
against the question, builds the generation prompt explicitly from the
ranked results (`build_rag_prompt`), and then calls a generation
`ModelAdapter` with that prompt. Nothing about retrieval is hidden inside
the model adapter, and nothing about generation is hidden inside the
retriever: each stage is a distinct object, callable and testable in
isolation, and the pipeline itself does nothing except compose them and
record what happened.

Because `RagPipeline` implements the existing `ModelAdapter` interface
from the Core Evaluation Engine (Prompt 1), it runs through the
unmodified `EvaluationRunner` and `ExperimentManager` (Prompts 1-2) with
no changes to either. Retrieval evidence -- the retrieved chunks, their
ranks and similarity scores, and the retrieval configuration -- travels
through `ModelResponse.provider_metadata["rag"]`, the extension point
`ModelResponse` was explicitly designed with for this purpose. Retrieval
ground truth travels through the existing `TestCase.metadata` under the
key `"relevant_chunk_ids"`. No persisted schema in
`llm_reliability.evaluation` or `llm_reliability.experiments` needed to
change to support RAG execution.

## Why Retrieval Must Be Evaluated Separately From Generation

A wrong final answer has at least two structurally different possible
causes: the retriever did not surface the information needed to answer
correctly, or the retriever did surface it and the generation step failed
to use it correctly. These require different fixes (a better retriever
or corpus, versus a better prompt or model), so conflating them into one
"the answer was wrong" signal destroys the information needed to act on
a failure. This is why `llm_reliability.rag.evaluators.RetrievalEvaluator`
and its `RetrievalEvaluationResult` are a completely separate type from
the answer-quality `Evaluator`/`EvaluationResult` (from
`llm_reliability.evaluation`), never merged into one score: a retrieval
score answers "did retrieval do its job," and an answer score answers
"was the final text correct," and only by keeping both can a later
experiment attribute a failure to the right stage.

## Retrieval Ground Truth

Not every test case has known relevant chunks -- most do not. A test
case's ground truth, if any, is a list of relevant chunk ids stored under
`TestCase.metadata["relevant_chunk_ids"]`
(`llm_reliability.rag.ground_truth.get_relevant_chunk_ids`). Its absence
(`None`) is a distinct, first-class state from an empty list or from
"zero relevant chunks were retrieved" -- every retrieval evaluator here
reports `SKIPPED` when ground truth is unavailable, never a fabricated
zero score. This mirrors the same "skipped, not failed" principle
`EvaluationStatus.SKIPPED` established in Prompt 3 for reference-based
answer evaluation.

## Retrieval Metrics, Precisely Defined

All three metrics are computed by `llm_reliability.rag.evaluators` and
require ground truth (relevant chunk ids for a test case) and retrieval
evidence (the ranked retrieved chunk ids from a RAG execution); both must
be available or the result is `SKIPPED`.

- **Hit@K**: 1.0 if at least one ground-truth relevant chunk id appears
  among the top K retrieved chunk ids, else 0.0. Answers: "did retrieval
  surface *any* relevant evidence at all in the top K?"
- **Recall@K**: `|relevant intersect top_k| / |relevant|` -- the fraction
  of *all* known relevant chunk ids that appear in the top K. Answers:
  "of everything relevant, how much did retrieval surface?" When a test
  case has exactly one relevant target, Hit@K and Recall@K are always
  numerically identical (both 1.0 if that one chunk is retrieved, else
  0.0); they diverge only when more than one relevant target exists,
  which is why the benchmark experiment for this milestone includes a
  "multiple relevant documents" case specifically to exercise that
  divergence.
- **Mean Reciprocal Rank (MRR)**: `1 / rank` of the first retrieved chunk
  that is in the ground-truth relevant set, or 0.0 if none are. Each
  evaluator invocation reports one query's reciprocal rank; the "mean" is
  computed separately, across many test cases, by
  `llm_reliability.rag.summary.summarize_retrieval`.

None of these metrics is computed when ground truth is absent, and none
of them substitutes for it with an invented relevance label.

## Context Support: What It Measures and What It Does Not

`ContextSupportBaseline` (`llm_reliability.rag.context_support`) reports
the fraction of an answer's unique word tokens that also appear in the
retrieved context. This is deliberately not called "faithfulness" or
"groundedness," and it is not a hallucination detector: literal word
overlap does not establish semantic entailment (an answer can repeat
context words while drawing an unsupported conclusion from them), and low
overlap does not establish contradiction (a correct paraphrase can share
few words with its source). It exists to establish the data flow -- an
answer paired with the context it was generated from -- that a later,
more rigorous faithfulness evaluator (NLI-based verification, claim
extraction, or an LLM-as-a-judge faithfulness rubric) will need. Reading
a high `ContextSupportBaseline` score as evidence of correctness, or a
low one as evidence of hallucination, is exactly the kind of overclaim
this platform's central principle warns against.

## Failure Diagnosis: Evidence-Based, Not a Guess

`llm_reliability.rag.diagnosis.diagnose_rag_execution` combines exactly
two signals -- a retrieval hit/miss signal (from Hit@K/MRR-style
ground-truth evaluation) and an answer-correctness signal (from
`criterion == CORRECTNESS` answer evaluation) -- into one of five
categories, by a fixed decision table:

| retrieval_hit | answer_correct | category |
|---|---|---|
| unavailable | any | `retrieval_evidence_unavailable` |
| True | True | `successful_grounded_execution` |
| True | False | `generation_failure_despite_relevant_context` |
| True | unavailable | `undetermined` |
| False | False | `retrieval_failure` |
| False | True | `undetermined` |

Two points are worth making explicit:

1. **An incorrect answer alone is never classified as retrieval
   failure.** `retrieval_failure` requires the additional, positive
   evidence that a known-relevant chunk was *not* retrieved. Without
   ground truth, the honest answer is `retrieval_evidence_unavailable`,
   not a guess.
2. **A correct answer despite a retrieval miss is `undetermined`, not
   success.** The required evidence was not surfaced (a real retrieval
   shortcoming) even though generation happened to produce a correct
   answer anyway -- for example, from the model's own parametric
   knowledge. This platform does not have, and will not invent, a
   category that quietly resolves that tension into "everything is
   fine."

When multiple retrieval or answer evaluators disagree with each other
(for example, one reports a hit and another does not), that disagreement
itself aggregates to "unavailable" evidence rather than being resolved by
arbitrarily picking one evaluator's opinion.

## Why an Incorrect Answer Does Not Automatically Imply Hallucination

Hallucination is a stronger, more specific claim than "wrong": an
assertion unsupported by, or contradicting, available context. Determining
that requires context-grounded verification -- checking a specific claim
in the answer against specific evidence in the context -- which this
milestone's `ContextSupportBaseline` explicitly does not attempt to do
rigorously (it measures overlap, not entailment). None of this module's
vocabulary uses the word "hallucination": `RagDiagnosisCategory`,
`ReliabilityFlagType` (Prompt 3), and `ContextSupportBaseline`'s labels
all describe evaluation evidence and its limits, never a factual-truth
verdict. A real hallucination detector -- NLI-based claim verification,
LLM-as-a-judge faithfulness rubrics, or equivalent -- is explicitly
deferred to a later milestone.

## Deliberately Excluded From This Milestone

- Semantic, hierarchical, or recursive chunking -- only fixed-size
  character chunking.
- A vector database or approximate nearest-neighbor index -- retrieval
  scores every candidate exactly, appropriate for the small, controlled
  corpora this milestone targets.
- NLI-based faithfulness verification, claim extraction, or LLM-as-a-judge
  faithfulness rubrics.
- A generic document ingestion system, multiple embedding providers, or a
  plugin architecture for retrieval components.
- Hallucination detection as a generic classifier.
