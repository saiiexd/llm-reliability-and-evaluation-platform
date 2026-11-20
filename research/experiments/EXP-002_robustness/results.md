# EXP-002 results — robustness to controlled perturbations (RQ5)

All numbers come from the persisted runs and `scripts/analyze_robustness_study.py`. The full tables
are in `results_tables.md`, the per-question answers in `per_case.csv`, and the archived runs in
`runs.tar.gz`. The design and hypotheses (`definition.md`) were committed before any run.

## 1. Execution record

| Attempt | When (UTC) | Outcome | Used as |
|---|---|---|---|
| 1 | 2026-11-19 16:56–17:16 | Ollama was OOM-killed twice by `systemd-oomd`, caused by other memory-heavy work on the same machine. 2 paraphrase generations failed with connection errors. | Kept in `attempt1_disrupted/` as a replication under degraded conditions |
| 2 | 2026-11-19 17:21–17:40 | All 6 runs answered 60/60; no OOM events in the journal | **Primary result** (below) |

The decision to repeat the study was made when the OOM kills were found, before the attempt-1 tables
were read. On the primary measure, attempt 1 gives **the same** flip counts as attempt 2 for the
noise floor, `reversed` and `distractor`, and close values for `paraphrase` (17.9% vs 19.3%) and
`freeform` (18.6% vs 16.7%).

## 2. Hypotheses vs results

Primary measure: share of questions whose LLM-judge verdict differs from `base` run 1, with 95%
Wilson intervals. The **noise floor** is the same measure for a second, unchanged `base` run.

| Contrast | n | Judge correct (ref → arm) | Flips (→ fail / → pass) | Flip rate [95% CI] | McNemar p |
|---|---|---|---|---|---|
| Noise floor (base run 2) | 60 | 35 → 36 | 0 / 1 | 1.7% [0.3, 8.9] | 1.00 |
| Reversed passage order | 60 | 35 → 38 | 0 / 3 | 5.0% [1.7, 13.7] | 0.25 |
| Irrelevant passage first | 60 | 35 → 33 | 3 / 1 | 6.7% [2.6, 15.9] | 0.63 |
| Reworded question (primary, n = 57) | 57 | 33 → 30 | 7 / 4 | 19.3% [11.1, 31.3] | 0.55 |
| Reworded question (all 60) | 60 | 35 → 32 | 7 / 4 | 18.3% [10.6, 29.9] | 0.55 |
| Free-form answer template | 60 | 35 → 39 | 3 / 7 | 16.7% [9.3, 28.0] | 0.34 |

| Hypothesis | Verdict | Evidence |
|---|---|---|
| H1: reversed order flips < 10%, net change not significant | **Supported** | 5.0%, p = 0.25. Retrieval was identical for all 60 questions, as designed. |
| H2: irrelevant passage flips < 10% | **Supported** | 6.7%. One answer became "I don't know"; one was shortened to a partial answer. |
| H3: rewording flips more than H1/H2 (≥ 10%) and changes retrieval | **Supported (descriptively)** | 19.3% vs 5.0% and 6.7%. The retrieved set changed for 42 of 57 questions; Hit@3 went from 86.7% to 83.3%. The intervals overlap with H1/H2 at their edges; a test of the difference between arms was not planned. |
| H4: free-form template drops EM and SQuAD EM by ≥ 15 points; judge changes by < 5 points | **Partly supported** | EM 30.0% → 1.7%, SQuAD EM 43.3% → 1.7% (McNemar p = 1.5e-5 and 6.0e-8). Judge **+6.7 points** (58.3% → 65.0%, p = 0.34): beyond the 5-point bound, though not significant. |

## 3. What the results show

**Observed:**
- With nothing changed, the judge verdict changed on 1 of 60 questions, and 56 of 60 outputs were
  identical. This is close to EXP-001's 57/60 identical outputs.
- **Reversing the passage order and adding an off-topic passage moved few verdicts.**
  - Reversing changed 14 of 60 outputs but only 3 verdicts, all towards correct.
  - Reversing also raised SQuAD EM from 26 to 32 correct (1 → fail / 7 → pass, p = 0.07). The best
    passage then sits next to the question. This is a possible position effect, not significant
    at n = 60.
- **Rewording the question changed retrieval for most questions and the verdict for about one in
  five.** The net effect was small and not significant (7 worse, 4 better). Wording sensitivity is
  large per question but not systematically harmful here.
- **The answer-format instruction dominates lexical metrics.**
  - With the free-form template, exact match and SQuAD EM fell to 1 of 60, because answers became
    sentences: 26 completion tokens on average vs 10.
  - The judge's correct rate did not fall. Its pass→fail and fail→pass flips (3 and 7) are within
    the range of the other perturbations.

**Interpretation (not measured):**
- **Some "flips" are judge inconsistency, not changes in correctness.** Several changed verdicts
  compare near-identical answers:
  - "soluble components (molecules)" vs the same phrase continued;
  - a list with and without a leading "from";
  - "Detention" vs a sentence containing it, with the judge passing the short form and failing the
    sentence.

  One question (`squad-56e75a9037bdd419002c3ec7`, "common punishment in Singapore") flips in
  **every** contrast, including the noise floor. Flip rates therefore mix answer change with judge
  sensitivity to answer form. EXP-003 measures the judge's own noise and a revised rubric on fixed
  answers.
- **The rule-based diagnosis is not robust to answer format.** It stays unchanged for only 35 of 60
  questions under the free-form template, mostly because the lexical evaluators fail on sentence
  answers and the case is then labelled `evaluator_disagreement`. This repeats EXP-001 issue P2.

## 4. Secondary measures

| Run | Exact | SQuAD EM | Judge correct | Faithful | Hit@3 | Precision@3 | MRR | Prompt / completion tokens |
|---|---|---|---|---|---|---|---|---|
| base 1 | 30.0% | 43.3% | 58.3% | 98.3% | 86.7% | 29.4% | 0.719 | 491 / 10.2 |
| base 2 | 30.0% | 43.3% | 60.0% | 98.2% | 86.7% | 29.4% | 0.719 | 491 / 10.5 |
| reversed | 33.3% | 53.3% | 63.3% | 98.2% | 86.7% | 29.4% | 0.719 | 491 / 9.7 |
| distractor | 23.3% | 41.7% | 55.0% | 98.2% | 86.7% | 29.4% | 0.719 | 607 / 9.9 |
| paraphrase | 26.7% | 35.0% | 53.3% | 100.0% | 83.3% | 27.8% | 0.706 | 496 / 8.8 |
| freeform | 1.7% | 1.7% | 65.0% | 98.2% | 86.7% | 29.4% | 0.719 | 466 / 26.0 |

- **Precision@3:** at most 33.3% here, because each question has one labelled chunk (1/K). 29.4%
  means the labelled chunk is in the top 3 for most questions. Read it with Hit@3.
- **Faithfulness:** near 100% in every arm, including abstentions. The v1 faithfulness judge
  labels "I don't know" `supported` (EXP-001 E1); EXP-003 tests rubric v2.
- **Latency is not interpreted.** The two identical `base` runs had median generation latencies
  of 354 ms and 183 ms. The first part of attempt 2 overlapped with test suites from another task
  on the same machine, which can explain this. Latency differences between arms therefore cannot be
  attributed to the perturbations. Correctness results are unaffected: no request failed, and the
  answers match attempt 1 closely.

## 5. Limitations

- **Judge:** one judge (qwen3:4b-instruct), the same model family as the generator. Its validity
  is unmeasured until human labels exist (`research/annotation/`).
- **Sample:** one generator, one dataset, n = 60. Differences below about 10 points between arms
  cannot be distinguished.
- **Paraphrases:** one per question, written by the generator's own model. Their meaning was
  reviewed by the AI assistant, not by a human (3 changed meaning and are excluded from the primary
  analysis).
- **Distractor:** a single kind, the least-similar passage. It is easy to ignore; related but
  wrong passages, as studied by Shi et al. (2023), were not tested.
