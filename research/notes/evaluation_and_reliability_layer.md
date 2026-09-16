# Evaluation and Reliability Layer

## Purpose of This Note

This note explains the concepts introduced by
`src/llm_reliability/evaluation/{criteria,normalization,semantic_similarity,llm_judge}.py`
and `src/llm_reliability/reliability/`: how an evaluation *criterion* differs
from an evaluation *method* (evaluator), how the three evaluators
implemented so far relate to each other, and how reliability analysis
aggregates their results without treating any one of them as ground truth
or collapsing an experiment into a single score.

## Criterion vs. Method vs. Evaluator vs. Result

- **Evaluation criterion** (`llm_reliability.evaluation.criteria`): a
  stable identifier for *what* is being assessed. This milestone defines
  exactly one, `CORRECTNESS` -- whether a generated answer is correct
  relative to a reference answer. It is deliberately just a string
  constant, not a class hierarchy: the goal is only that multiple
  evaluators can declare they assess the same thing, so an experiment can
  compare their verdicts for that criterion.
- **Evaluation method / evaluator** (`Evaluator` subclasses): *how* a
  criterion is assessed. Three exist so far, all assessing correctness by
  different methods: `ExactMatchEvaluator` (string identity),
  `SemanticSimilarityEvaluator` (embedding similarity), and
  `LLMJudgeEvaluator` (a judge model applying an explicit rubric). Each
  evaluator has a stable `name` (identity) and a `criterion` (what it
  assesses); `Evaluator.get_config()` returns its serializable
  configuration, which becomes part of the experiment's reproducibility
  fingerprint (see `experiment_reproducibility_model.md`) automatically,
  with no change needed to the experiment system itself.
- **Evaluation result** (`EvaluationResult`): the structured, per-test-case
  output of running one evaluator. Its `status` field
  (`EvaluationStatus`) is the most important addition this milestone
  makes: it distinguishes `SUCCESS` (a definitive judgment was produced)
  from `SKIPPED` (required input, such as a reference answer, was
  unavailable for this test case -- not a failing judgment),
  `INVALID_CONFIGURATION` (the evaluator cannot function as configured in
  this environment, for example a missing model dependency),
  `EXECUTION_ERROR` (an unexpected runtime failure, including a failed
  judge model call), and `OUTPUT_VALIDATION_ERROR` (a judge's response
  could not be parsed into the rubric's required structure). A `None`
  score or `passed` value is never itself evidence of a wrong answer; it
  means no judgment was reached, and `status` says why.
- **Reliability analysis** (`llm_reliability.reliability`): a read-only
  layer that classifies the evaluation evidence already produced --
  aggregated per evaluator (`summarize_run`) and flagged per test case
  (`analyze_reliability`). It never re-executes anything and never
  produces a new judgment of its own.

## Why No Single Metric Is Ground Truth

The project's central principle (`research/project_definition.md`) is that
automated evaluation methods must themselves be assessed for reliability,
not assumed correct. Each evaluator here has a specific, limited
methodology:

- **Exact match** (`ExactMatchEvaluator`) is a strict string-identity
  baseline. Its normalization is explicit and narrow -- optional
  whitespace collapsing and case folding, nothing else (no punctuation
  removal, stemming, or synonym handling; see
  `llm_reliability.evaluation.normalization`) -- so its behavior is fully
  predictable. It exists to validate the execution and evaluation
  infrastructure end to end, not to measure answer quality: it cannot
  recognize a correct paraphrase, and a reference answer that is only one
  of several valid answers (see the benchmark's `ambiguous_or_insufficient_reference`
  category) will make exact match report a "mismatch" for other equally
  correct answers.
- **Semantic similarity** (`SemanticSimilarityEvaluator`, following
  BERTScore -- Zhang et al., 2020) measures contextual embedding
  similarity rather than string identity, and so can recognize
  paraphrase. This complements exact match, but high similarity does
  **not** establish factual correctness, completeness, or faithfulness:
  two answers can be highly similar in wording while one is factually
  correct and the other is not (for example, a paraphrase with a single
  number changed). The evaluator's real backend (`BertScoreBackend`) is
  isolated so it can be tested and configured without the `bert-score`
  dependency or a model download; see "BERTScore Availability" below.
- **LLM-as-a-judge** (`LLMJudgeEvaluator`), following the LLM-as-a-judge
  approach described by Zheng et al., 2023 ("Judging LLM-as-a-Judge with
  MT-Bench and Chatbot Arena"), asks a judge model to apply an explicit,
  versioned rubric (`JudgeRubric`) rather than an unconstrained scoring
  request. This is itself an evaluation *method*,
  not a source of ground truth: its output reflects the judge model's own
  behavior, training, and biases, and its agreement with human judgment
  must be established separately (a later milestone) before its verdicts
  can be trusted as a substitute for human review. This platform does not,
  and must not, treat a judge's output as human truth.

Because these three methods can and do disagree (see the benchmark's
`paraphrasable` category, where an evaluator run intentionally produces
such disagreement), the platform preserves every evaluator's result
individually rather than averaging or voting them into one score. See
"HELM and Multi-Metric Evaluation" below for why this design choice
follows established practice in the field.

## BERTScore Availability

`bert-score` (and its `torch`/`transformers` dependencies) is declared as
an optional `semantic` extra (`pip install ".[semantic]"`), not a core
dependency: `BertScoreBackend` imports it lazily, only inside
`compute_similarity`, so constructing and configuring
`SemanticSimilarityEvaluator` never requires it. In this development
environment, `bert-score` itself is not installed (only its underlying
`torch`/`transformers` dependencies happen to be present for unrelated
reasons), and a real BERTScore call would download a pretrained model on
first use, which requires network access this project's standard test
suite must not depend on. The standard test suite therefore exercises the
evaluator's full contract -- configuration, thresholding, skip behavior,
error handling -- against `FakeSimilarityBackend`, a deterministic fake
that performs no embedding computation at all and must never be read as
evidence about real BERTScore behavior. `tests/evaluation/test_semantic_similarity_integration.py`
exercises the real backend and is skipped automatically unless
`bert-score` is installed; it asserts only structural correctness and
directionally expected behavior (a paraphrase should score above an
unrelated sentence), not fixed numerical thresholds, since exact BERTScore
values depend on the installed model version.

## HELM and Multi-Metric Evaluation

This platform's decision to preserve multiple, disagreeing evaluator
results per test case rather than reducing to one score follows the
methodological stance of HELM (Holistic Evaluation of Language Models,
Liang et al., 2022): that no single metric captures model quality, and
that making multiple metrics' disagreement visible is itself valuable
information, not noise to be averaged away. `EvaluationSummary` reflects
this directly: it has no "overall score" field, only per-evaluator
statistics, each explicitly labeled with the evaluator and criterion it
belongs to.

## What Reliability Analysis Does and Does Not Do

`llm_reliability.reliability.analyze_reliability` currently identifies
four things, all fully grounded in evaluation results already computed:

1. **Evaluator disagreement** -- evaluators that succeeded reached
   different pass/fail verdicts for the same test case.
2. **Missing evaluation evidence** -- every evaluator was skipped for a
   test case (for example, no reference answer was available for any
   reference-based method).
3. **Evaluator failure** -- at least one evaluator could not produce a
   result (`INVALID_CONFIGURATION`, `EXECUTION_ERROR`, or
   `OUTPUT_VALIDATION_ERROR`).
4. **Incorrect per evaluator** -- a specific evaluator reported the answer
   as not passing, relative to that evaluator's own methodology.

None of these is a hallucination determination, and the word
"hallucination" does not appear as a label anywhere in this layer's
output. Hallucination is a stronger, more specific claim -- an assertion
unsupported by, or contradicting, available context -- that requires
context-grounded evaluation (checking an answer against retrieved or
provided source material, not just a single reference string). That
requires a retrieval/context pipeline this platform does not yet have,
and is explicitly deferred, along with faithfulness, groundedness, and
retrieval failure analysis, to a later milestone.

## Deliberately Excluded From This Milestone

- RAG execution, retrieval, or context-grounded faithfulness evaluation.
- Hallucination detection as a generic classifier.
- Robustness/consistency experiments (paraphrase perturbation, repeated
  sampling) -- this milestone provides the multi-evaluator machinery such
  experiments will use, but performs none of them.
- A frontend, dashboard, or API layer.
- Validating LLM-as-a-judge's reliability against human annotation; that
  is future work this note explicitly does not claim has happened.
