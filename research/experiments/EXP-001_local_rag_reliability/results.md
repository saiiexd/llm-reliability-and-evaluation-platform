# EXP-001 — Results, interpretation and limitations

Run: 2026-11-19, 9 runs × 60 questions = 540 generations, 0 execution errors. Exact numbers,
intervals and environment are in [`results_tables.md`](results_tables.md) (generated) and
[`results.json`](results.json). Every answer and verdict is in [`per_case.csv`](per_case.csv).
Design and hypotheses: [`definition.md`](definition.md) (written before the run).

**Raw runs.** All 9 persisted runs (prompts, retrieved passages, outputs, every evaluator result,
provenance) are archived in [`runs.tar.gz`](runs.tar.gz) (0.6 MB). To load them into the console
and re-run the analysis:

```bash
tar xzf research/experiments/EXP-001_local_rag_reliability/runs.tar.gz -C experiments
python scripts/analyze_reliability_study.py
```

**Observed** marks a measurement. **Interpretation** marks what we think it means.

## 1. Headline numbers (first run of each arm, n = 60, 95% Wilson intervals)

| Arm | Exact match | SQuAD EM | SQuAD F1 | LLM judge "correct" | Abstained |
|---|---|---|---|---|---|
| qwen3:4b closed book | 5.0% [1.7, 13.7] | 6.7% [2.6, 15.9] | 0.19 | 13.3% [6.9, 24.2] | 25.0% |
| qwen3:4b top-1 | 23.3% [14.4, 35.4] | 36.7% [25.6, 49.3] | 0.51 | 50.0% [37.7, 62.3] | 23.3% |
| qwen3:4b top-3 | 30.0% [19.9, 42.5] | 43.3% [31.6, 55.9] | 0.65 | 58.3% [45.7, 69.9] | 10.0% |
| gemma3:1b closed book | 3.3% [0.9, 11.4] | 6.7% [2.6, 15.9] | 0.13 | 6.7% [2.6, 15.9] | 1.7% |
| gemma3:1b top-3 | 25.0% [15.8, 37.2] | 43.3% [31.6, 55.9] | 0.58 | 50.0% [37.7, 62.3] | 3.3% |

Retrieval (nomic-embed-text, 378 chunks; strict source-paragraph labels):

- **Hit@1:** 60.0%.
- **Hit@3:** 86.7%, MRR 0.72.
- **Lenient Hit@3:** 91.7% (any retrieved chunk contains a gold answer string).

## 2. Hypotheses

| | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| H1 | Top-3 retrieval improves qwen over closed book | **Supported** | Judge 8 → 35 of 60. Paired: 29 cases only RAG correct vs 2 only closed-book correct; exact McNemar p = 4.6e-7. Same direction for gemma (p = 2.2e-7) and with SQuAD EM. |
| H2 | Top-3 ≥ top-1 | **Not demonstrated** | Judge 30 → 35, discordant 8 vs 3, p = 0.23. The direction favours top-3, but n = 60 cannot separate it from noise. |
| H3 | qwen3:4b beats gemma3:1b with top-3 | **Not demonstrated** | Judge 35 vs 30, discordant 9 vs 4, p = 0.27. SQuAD EM is exactly tied, 26 vs 26, with 9 discordant each way. The judge is qwen itself, so even this gap is confounded by self-preference. |
| H4 | Exact match underestimates; SQuAD narrows the gap; agreement at most moderate | **Supported** | qwen top-3: EM 30%, SQuAD EM 43%, judge 58%. Kappa EM vs judge 0.44–0.50 across RAG arms ("moderate"); SQuAD vs judge 0.42–0.60. |
| H5 | T = 0 runs are identical; T = 0.8 changes outputs but few verdicts | **Partly rejected** | T = 0 with a fixed seed: 57/60 identical. 3 answers differed in length, e.g. "Detention" vs a full sentence. T = 0.8: 50/60 identical across 3 runs; judge verdict changed on 1 question; pass rate 60.0 / 58.3 / 58.3%. |

## 3. What the results show

**Retrieval is the dominant factor here.** Observed:

- Both models are nearly useless closed-book on these questions: 7–13% by the judge.
- Both models jump to 50–58% with top-3 retrieval.

Interpretation: SQuAD questions were written while looking at one paragraph. Many are
under-specified without it ("What did the museum acquire from …"), so the closed-book arm is a
floor rather than a fair test of knowledge. This experiment can't separate "the model doesn't know"
from "the question is unanswerable without the paragraph".

**The answer depends strongly on whether the right passage was retrieved.** For qwen top-3, the
judge marked 65% of hit cases correct vs 12.5% of the 8 miss cases. For top-1: 81% of hits vs
4% of 24 misses. Interpretation: a large share of RAG failures here are retrieval failures, not
generation failures.

Hit cases in the top-1 arm are more often correct than hit cases in the top-3 arm (81% vs 65%).
Observed, but these are different subsets: top-1 hits are the easy cases where the answer passage
ranked first. It is not evidence that extra passages distract the model.

**The 4B model abstains; the 1B model guesses.** Closed-book, qwen said "I don't know" on 25% of
questions and gemma on 1.7%, with the same instruction. Gemma's extra answers were nearly all
wrong (judge 6.7%). Interpretation: under the same accuracy, the larger model is better calibrated
about what it does not know. This is the kind of reliability difference a single accuracy number
hides.

**Evaluator choice changes the conclusion more than model choice does.** For the same 60 qwen top-3
answers, "accuracy" is 30%, 43% or 58% depending on the evaluator. That spread (28 points) is larger
than every model or retrieval-depth difference measured. Exact match never passed an answer the
judge failed (0 cases in the "EM only" cell in every arm), so *relative to the judge* its errors are
all false negatives (the judge itself is not yet validated against humans).

**The judge is not uniformly lenient.** In 4 qwen top-3 cases SQuAD EM passed but the judge did
not; in 13 the judge passed what SQuAD EM failed (mostly correct but longer answers). Spot checks show the judge marking "the Henry Cole wing" vs gold "Henry Cole wing", and "pharma"
vs gold "its root word pharma", as `partially_correct`. The judge sees only the first gold answer,
and `partially_correct` counts as fail. These are judge errors, not model errors.

**The faithfulness judge has little discriminating power on qwen.** It labelled 57/58 qwen top-3
answers `supported`, and all 8 abstentions ("I don't know") in both RAG arms as `supported`. An
abstention makes no claim; labelling it "supported" inflates faithfulness. Gemma produced more
`contradicted` (7) and `undetermined` (6) labels, consistent with it guessing.

**Measured run-to-run variation is small but non-zero at T = 0.** With temperature 0 and a fixed
seed, 3/60 outputs changed between runs, so configuration reproducibility did not give execution
determinism, even locally. At T = 0.8, 10/60 outputs changed across three runs, but only one judge
verdict changed: most variation was wording, not correctness.

**Rule-based diagnosis is dominated by evaluator disagreement.** About a third of cases in RAG arms
(18–22 of 60) are labelled `evaluator_disagreement`, because exact match and the judge disagree on
correct-but-verbose answers. Interpretation: the diagnosis rule "correctness evaluators disagree →
disagreement" is too coarse when one evaluator is a known-strict formatting check. Retrieval
failures (7 at k = 3, 23 at k = 1) and generation failures (13–14) are the next largest groups.

**Operational (RQ7).** Median generation latency on an RTX 3050 6 GB:

- 55 ms for gemma closed-book, 350 ms for qwen top-3.
- Prompt size grows from about 65 to 500 tokens with three passages.
- One cold model load remained in gemma-closed (max 2.2 s); medians and p90 are unaffected. The runner now warms up before every arm.

## 4. What cannot be concluded

- **Anything about human-judged correctness.** No human labels exist yet. Treat "judge correct" as one evaluator's opinion.
- **Model ranking.** qwen vs gemma is not significant, and the judge is qwen (self-preference confound).
- **General statements about these models.** The data is one dataset (6 SQuAD articles), one prompt per condition, one chunking setting, one embedding model and n = 60 per arm. Intervals are about ±12 points.
- **Knowledge vs memorization.** SQuAD is public; closed-book scores may partly reflect memorization.
- **Significance correction.** Four contrasts × two evaluators were tested without correction. Only the retrieval effect (p < 1e-5) would survive any reasonable correction.

## 5. Next experiments (follow directly from these results)

1. **EXP-002 judge validity.** Two people label the 360 first-run answers in `per_case.csv`
   (`human_correct` column). Then report judge, SQuAD and EM accuracy vs humans and inter-annotator
   kappa. Also give the judge all gold answers, and treat `partially_correct` separately.
2. **Faithfulness rubric v2.** Add an explicit `no_claim` / abstention outcome. Re-judge the
   existing answers; the runs are stored, so no regeneration is needed.
3. **Neutral judge.** Re-judge with a model that is not one of the compared generators before
   making any model-comparison claim.
4. **Diagnosis rule.** Designate a primary correctness evaluator (or exclude formatting-only
   evaluators from the disagreement rule) and measure how category counts change.
5. **Power.** For retrieval depth and model differences of about 5–10 points, several hundred
   questions are needed, not 60.
