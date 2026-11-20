# EXP-003 — Does judge rubric v2 fix the judge problems found in EXP-001?

Status: defined 2026-11-19, **before any v2 judgement was produced**. The rubrics were written
from the EXP-001 failure analysis (findings E1 to E3), without looking at v2 outputs.

## Research question

**RQ3 — Evaluator reliability.** EXP-001 found three problems with the LLM judges. Does a revised
rubric change them, on the *same* answers?

| # | Problem in v1 | v2 change |
|---|---|---|
| E1 | The faithfulness judge labelled 8/8 abstentions ("I don't know") `supported` | New `no_claim` outcome for answers that make no claim; scored as no verdict |
| E2 | The correctness judge saw only the first gold answer and marked surface variants `partially_correct` | v2 lists every accepted answer and states rules for surface variants |
| E3 | One judge output was truncated (invalid JSON) | v2 asks for one-sentence reasoning |

## Design

- **No new generation.** `ReplayAdapter` re-serves the recorded answers of existing runs, so any
  verdict change comes from the judge, not the answer.
- **Fixed for every arm:**
  - judge model qwen3:4b-instruct;
  - temperature 0, seed 42, `max_tokens = 200`, JSON response format, same as EXP-001.
- **Source runs (answers are fixed):**

  | Source | Why |
  |---|---|
  | EXP-001 `qwen-rag-k3` run 1 | Main RAG arm; contains abstentions |
  | EXP-001 `gemma-rag-k3` run 1 | A second, weaker generator |
  | EXP-001 `qwen-closed` run 1 | Many abstentions (25%); correctness only |
  | EXP-002 `freeform` run 1 | Sentence-length answers; tests E2's surface-variant rules |

- **Arms per source:**
  - **`v1`:** re-judge with the original v1 rubrics. Compared with the verdicts originally
    recorded, this measures **judge noise** on identical inputs.
  - **`v2`:** judge with the v2 rubrics.
- **Evaluators per arm:**
  - `llm_judge` and `llm_faithfulness_judge` in the arm's version;
  - `squad_match`, deterministic, as the lexical reference.

## Hypotheses (stated before the run)

- **H1 (E1):** under v2, 0 abstentions are labelled `supported`. They are labelled `no_claim`.
  Fewer than 5% of non-abstaining answers are labelled `no_claim`.
- **H2 (E2):**
  - v2 changes correctness verdicts mostly fail→pass;
  - its pass→fail changes are no more frequent than the v1 judge-noise flips;
  - Cohen's κ between the judge and SQuAD EM is higher under v2 than under v1 on the same answers.
- **H3 (E3):** v2 has no more `output_validation_error`s than v1.

## Measures and analysis

- **Abstentions:** counted with the EXP-001 normalization ("i dont know"). Faithfulness labels are
  cross-tabulated by abstention for v1 and v2.
- **Judge noise:** flips between the original v1 verdict and the re-judged v1 verdict.
- **Rubric effect:** v1-rejudge vs v2 flips, split by direction, with an exact McNemar test.
  Each changed case is listed with the answer, the gold answers and both labels, for inspection.
- **Agreement with SQuAD EM:** raw agreement and κ for v1 and v2.
- **Judge failures:** `output_validation_error` and `execution_error` counts.

## Validity limits (stated in advance)

- There are **no human labels**. "Agrees more with SQuAD EM" is not the same as "more correct":
  SQuAD EM is itself a lexical proxy. Showing that v2 is *more valid* needs the human-annotation
  study (`research/annotation/`). This experiment can show only that v2 changes the targeted
  behaviours, and how much of the change exceeds judge noise.
- **Self-preference:** the judge is the same model that produced the qwen answers.
- **Sample:** n = 60 per source. Abstention counts are small.
