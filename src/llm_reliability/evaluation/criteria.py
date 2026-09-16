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
