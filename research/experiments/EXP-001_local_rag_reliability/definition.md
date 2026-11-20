# EXP-001 — Retrieval, model size and evaluator choice on a SQuAD subset (local models)

Status: defined 2026-11-19 before the full run; pilot of 5 questions used only to check the
pipeline (pilot results are not part of the analysis).

## Research questions addressed

| RQ | Question in this experiment |
|---|---|
| RQ1 model reliability | Does a 4B model answer more questions correctly than a 1B model under identical retrieval? |
| RQ2 retrieval effectiveness | Does adding retrieved context change correctness compared with the same model answering closed-book? Does top-3 differ from top-1? Is correctness higher when the relevant passage was retrieved? |
| RQ3 evaluator reliability | How often do exact match, SQuAD-normalized match and an LLM judge agree on the same answers (raw and chance-corrected)? |
| RQ4 output consistency | Are outputs identical across two temperature-0 runs? How often do outputs and verdicts change across three temperature-0.8 runs? |
| RQ6 failure attribution | How are wrong answers distributed over the platform's rule-based categories? |
| RQ7 operational performance | Generation latency and token usage per arm. |

RQ5 (robustness to prompt/context perturbation) is **not** addressed here.

## Hypotheses (stated before the run)

- H1: For qwen3:4b, top-3 retrieval increases LLM-judge correctness over closed book
  (SQuAD questions were written against a specific paragraph; many are under-specified without it).
- H2: top-3 retrieval is at least as good as top-1 (more chances to include the source paragraph),
  with possible distraction from irrelevant passages.
- H3: qwen3:4b is more often correct than gemma3:1b with the same top-3 retrieval.
- H4: Exact match underestimates correctness relative to the LLM judge (format strictness), and
  SQuAD normalization narrows but does not close the gap. Agreement beyond chance is moderate at best.
- H5: Two temperature-0 runs with a fixed seed produce identical outputs on this local server;
  temperature 0.8 produces some output changes but few verdict changes.

## Design

| Arm | Generator | Retrieval | Temperature | Seed | Runs |
|---|---|---|---|---|---|
| qwen-closed | qwen3:4b-instruct | none (closed book) | 0 | 42 | 1 |
| qwen-rag-k1 | qwen3:4b-instruct | top-1 | 0 | 42 | 1 |
| qwen-rag-k3 | qwen3:4b-instruct | top-3 | 0 | 42 | 2 |
| gemma-closed | gemma3:1b | none | 0 | 42 | 1 |
| gemma-rag-k3 | gemma3:1b | top-3 | 0 | 42 | 1 |
| qwen-rag-k3-t08 | qwen3:4b-instruct | top-3 | 0.8 | none | 3 |

**Independent variables:** retrieval (none / top-1 / top-3), generator model, temperature.

**Controlled variables (identical in every arm):** dataset version (same 60 questions, same
`relevant_chunk_ids`), corpus (266 paragraphs) and chunking (800 characters, 100 overlap,
378 chunks), embedding model (`nomic-embed-text`, with its `search_document:` / `search_query:`
prefixes), answer-format instruction ("reply with only the answer as a short phrase … or exactly
I don't know"), `max_tokens = 64`, the four evaluators and their configuration, the judge model and
its settings (qwen3:4b-instruct, temperature 0, seed 42, JSON response format), Ollama server
version and model digests (recorded in every experiment's metadata).

**Dependent variables:** per-evaluator pass rate; SQuAD token F1; faithfulness labels; Hit@K,
Recall@K, MRR (strict) and answer-containment Hit@K (lenient); evaluator agreement and Cohen's
kappa; rule-based failure categories; output identity across runs; generation latency and tokens.

**Baselines:** the closed-book arm of each model is the baseline for the retrieval effect;
qwen-rag-k1 is the baseline for retrieval depth.

## Dataset

`data/benchmarks/squad_v1_1_dev_subset_v1.json`, built by `scripts/build_squad_subset.py` from
the SQuAD v1.1 development set (Rajpurkar et al., 2016; CC BY-SA 4.0) with seed 2026:
6 articles chosen at random, all 266 of their paragraphs used as the corpus, 10 questions per
article from 10 distinct paragraphs (60 questions). All gold answers are kept.

**Relevance labels.** A chunk is relevant when it fully contains the first gold answer span of the
question's source paragraph (derived after chunking, not assumed). This is *strict*: another
paragraph can also contain the answer and would count as a miss. A *lenient* answer-containment
hit (any retrieved chunk contains any normalized gold answer string) is reported alongside it;
it over-counts for short, common answers.

## Evaluators

| Evaluator | Criterion | Definition |
|---|---|---|
| `exact_match` | correctness | Output equals the first gold answer after case and whitespace normalization only. |
| `squad_match` | correctness | SQuAD official normalization (lowercase, strip punctuation and a/an/the); pass = equals any gold answer; score = max token F1. |
| `llm_judge` | correctness | qwen3:4b judges the answer against the first gold answer with rubric `correctness_vs_reference_v1`; pass only for `correct` (not `partially_correct`). |
| `llm_faithfulness_judge` | faithfulness | qwen3:4b judges whether retrieved context supports the answer (`faithfulness_vs_context_v1`); skipped for closed-book arms. |

## Statistical analysis plan

- Pass rates with 95% Wilson intervals; n = cases the evaluator actually scored.
- Paired contrasts on the same 60 questions with the exact McNemar test (two-sided), for:
  qwen-closed vs qwen-rag-k3, qwen-rag-k1 vs qwen-rag-k3, gemma-closed vs gemma-rag-k3,
  gemma-rag-k3 vs qwen-rag-k3. Four contrasts × two evaluators; no multiple-comparison correction
  is applied, so p-values near 0.05 are treated as weak evidence.
- Agreement: raw agreement and Cohen's kappa per pair of correctness evaluators.

## Known threats to validity (stated in advance)

1. **Self-judging.** The judge (qwen3:4b) is the same model as the generator in four arms. LLM judges
   are known to favour their own outputs (self-preference bias), so qwen-vs-gemma differences
   judged by qwen are confounded.
2. **Judge validity is not established.** No human labels; the judge is an evaluation method, not
   ground truth.
3. **Small n.** 60 questions per arm: intervals are wide (about ±12 points at 50%).
4. **Training-data exposure.** SQuAD is public; closed-book answers may reflect memorization.
5. **One prompt template per condition**, one chunking setting, one embedding model.
6. **Local hardware** (RTX 3050 6 GB laptop GPU) drives latency; numbers do not transfer to other hardware.

## Reproduction

```bash
ollama pull qwen3:4b-instruct && ollama pull gemma3:1b && ollama pull nomic-embed-text
PYTHONPATH=. .venv/bin/python scripts/run_reliability_study.py        # writes to ./experiments
PYTHONPATH=. .venv/bin/python scripts/analyze_reliability_study.py    # writes results.json, results_tables.md
```

Model digests, Ollama version and the code commit are stored in each experiment's `metadata.study`.
Configuration reproducibility is not execution determinism: the same configuration can produce
different outputs if the server, drivers or model files change.
