# EXP-001 — Failure analysis

Problems found while running and reading EXP-001: in the evaluated systems, in the evaluators,
and in the platform. Each item has evidence and a proposed fix.

## Evaluator failures

| # | Problem | Evidence | Consequence | Proposed fix |
|---|---|---|---|---|
| E1 | Faithfulness judge labels abstentions `supported` | 8/8 "I don't know" answers in RAG arms | Faithfulness rate inflated; an abstention is counted as a supported claim | Rubric v2 with a `no_claim` outcome (passed = None) |
| E2 | Correctness judge is stricter than the gold set on surface variants | "the Henry Cole wing" vs "Henry Cole wing", "pharma" vs "its root word pharma" → `partially_correct` (4 such cases in qwen top-3) | Judge false negatives; `partially_correct` counts as fail | Give the judge all gold answers; report `partially_correct` separately |
| E3 | Judge output truncated by `max_tokens = 200` | 1 `output_validation_error` (qwen top-1): unterminated JSON string in a long reasoning | The case is "not scored" for faithfulness (correctly not a pass or fail) | Raise judge `max_tokens`, or ask for one-sentence reasoning |
| E4 | Exact match only produces false negatives here | "EM pass, judge fail" = 0 in every arm; 15–18 "judge pass, EM fail" per RAG arm | Any-fail case verdict and diagnosis are dominated by formatting | Use SQuAD EM / F1 as the lexical baseline for extractive QA; keep EM only as a formatting check |
| E5 | Self-judging | Judge = qwen3:4b, also the main generator | Model comparison confounded | Neutral judge (EXP-002/004) |

## System-under-test failures (observed, interpretation separated)

- **Retrieval misses dominate wrong answers at k = 1.** 24 misses; 23 of them are diagnosed as
  `retrieval_failure`, and the judge marked only 1 of the 24 correct. At k = 3, misses drop to 8.
- **Generation failures with the right passage retrieved.** 14 (qwen top-3), 13 (gemma top-3). In the
  spot-checked qwen cases the model usually gave a different span from the same passage, or an
  over-long answer. See `per_case.csv`, filter `relevant_chunk_ids` ⊂ `retrieved_chunk_ids` and
  judge ≠ correct.
- **Gemma guesses instead of abstaining.** Closed book: 1.7% abstentions vs qwen 25%, with near-zero
  accuracy for both.

## Platform issues surfaced by the experiment

| # | Issue | Status |
|---|---|---|
| P1 | First recorded call of a later arm included a model load (Ollama unloads idle models after 5 min) | Fixed in the runner (warm-up before every arm); one outlier remains in this run's gemma-closed max latency |
| P2 | `evaluator_disagreement` dominates the diagnosis (18–22 of 60 in RAG arms) because EM is in the disagreement rule | Open: designate a primary correctness evaluator (results.md §5) |
| P3 | Strict relevance labels miss answer-bearing paragraphs other than SQuAD's source paragraph (lenient Hit@3 91.7% vs strict 86.7%) | Reported both; open: graded relevance labels |
| P4 | Real judges are only available from the study script, not from the console | Open (technical report §G) |
| P5 | T = 0 with a fixed seed is not bit-for-bit repeatable on this server (3/60 outputs differ) | Documented; record outputs, never assume regeneration reproduces them |
