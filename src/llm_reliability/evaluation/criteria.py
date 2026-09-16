"""
Evaluation criteria: stable identifiers for what is being assessed.

A criterion describes *what* is being assessed (for example,
"correctness"). An evaluator (see ``evaluators.py``, ``semantic_similarity.py``,
``llm_judge.py``) describes *how* it is assessed (for example, exact string
match, semantic similarity, or LLM-as-a-judge). The same criterion can be
assessed by multiple evaluators; this is what lets an experiment compare
methods against each other for the same criterion rather than treating any
one evaluator as ground truth for it.

This is intentionally just a set of stable string identifiers, not a class
hierarchy, ontology, or plugin registry. Add a new constant here only when
a new evaluator actually needs to declare a criterion that does not already
exist.
"""

CORRECTNESS = "correctness"
"""Whether the generated answer is correct relative to a reference answer."""

CONTEXT_SUPPORT = "context_support"
"""Whether the generated answer has textual overlap with retrieved context.

This is deliberately narrower than "faithfulness" or "groundedness": it
measures literal word overlap, not semantic entailment, and is not a
hallucination detector. See
``llm_reliability.rag.context_support.ContextSupportBaseline``.
"""

FAITHFULNESS = "faithfulness"
"""Whether the generated answer is supported by, unsupported by, or
contradicted by retrieved context, as judged by a method capable of more
than literal word overlap (currently an LLM judge; see
``llm_reliability.rag.faithfulness_judge.LLMFaithfulnessEvaluator``). This
is the criterion for the full four-way assessment (supported / unsupported
/ contradicted / undetermined); ``CONTEXT_SUPPORT`` above remains the
narrower, lexical-overlap-only baseline for the same underlying question.
"""
