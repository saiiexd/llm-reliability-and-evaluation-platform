# LLM Reliability & Evaluation Platform — Technical Understanding Report

Date: 2026-11-19 · Branch: `fix/evaluation-integrity` · Author: repository audit (Claude Code) for the capstone team

Every statement is tagged:

- **[Verified]**: checked by reading the code and running it (tests, API probes, browser).
- **[Documented]**: stated in code comments or docs, not independently tested.
- **[Recommendation]**: an opinion about what to do next.

---

## A. The research problem (interpretation)

An LLM application can fail in several separable ways:

- **Retrieval** finds the wrong evidence.
- **Generation** ignores or contradicts good evidence.
- **The answer** is right but phrased so that a strict metric rejects it.
- **The evaluator** itself is wrong.
- **The call** crashes.

A single accuracy number mixes all five. The platform's purpose is to keep these stages separate and inspectable. For every case it records what was asked, retrieved and generated, and what each evaluator concluded and why. Diagnoses are attached as rule-based labels over that evidence. Runs are compared only when their configurations make the comparison meaningful.

The research contribution is therefore methodological rather than a new metric:

- a reproducible record of each run;
- separation of execution status from evaluation verdicts;
- evidence-linked failure attribution;
- explicit reporting of evaluator disagreement.

The project does not need to claim it is the first platform to do any of this. RAGAS, DeepEval, TruLens, MLflow Evaluate and HELM cover metrics and tracking. Its value is in how carefully the evidence is kept and interpreted on small, controlled experiments.

## B. Actual architecture (from the code)

**[Verified]** Python 3.13 package `src/llm_reliability`, FastAPI API, React 19 + Vite + Tailwind 4 console, JSON-file persistence.

```
evaluation/     domain models (TestCase, Dataset, ModelConfig, ModelResponse, EvaluationResult,
                TestCaseResult, RunResult), EvaluationRunner, evaluators (exact_match, squad_match,
                semantic_similarity, llm_judge), adapters (MockAdapter), prompting (closed-book template)
rag/            documents + FixedSizeChunker, embeddings (Fake, SentenceTransformer, OpenAI-compatible),
                Retriever (cosine), RagPipeline (is a ModelAdapter), retrieval evaluators (Hit@K,
                Recall@K, MRR), faithfulness judge, lexical context-support baseline, RAG diagnosis
experiments/    ExperimentDefinition / ExperimentRun, SHA-256 fingerprints, ExperimentManager,
                LocalFileExperimentRepository (atomic write: temp file + fsync + os.replace)
providers/      BaseProviderAdapter (opt-in guard, bounded retries with backoff, latency, token
                provenance), OpenAIProviderAdapter (OpenAI and OpenAI-compatible: Ollama, vLLM)
diagnostics/    taxonomy, consolidated diagnosis, sentence-level lexical evidence, robustness pairs
reliability/    per-evaluator summaries, reliability findings (evidence flags, not model claims)
analytics/      aggregations, run comparison + compatibility, evaluator agreement (+ Cohen's kappa),
                trends
api/            app factory, allowlisted adapter/evaluator registries, routers (health, providers,
                experiments, runs, diagnostics, rag, robustness, analytics, benchmarks), schemas,
                request-correlation middleware, error envelope
frontend/src    four sections: Overview, Experiments, Results, Analysis (tabs: reliability,
                comparison, retrieval, robustness, performance); live API or bundled fixtures
```

Main design decisions, each **[Verified]**:

- **`RagPipeline` is just another `ModelAdapter`.** The runner and experiment system need no RAG-specific code. Retrieval evidence travels in `ModelResponse.provider_metadata["rag"]`, together with the exact prompt.
- **Evaluators are independent and record their own configuration.** That configuration is part of the experiment fingerprint.
- **Fixed allowlists.** The API can only build adapters and evaluators from allowlisted registries; there is no dynamic import from requests.
- **No database.** Persistence is plain JSON files.

## C. End-to-end lifecycle of one experiment (traced in code)

| Step | Where | What happens |
|---|---|---|
| 1. Definition | `POST /experiments` → `routers/experiments.py:create_experiment` | Request validated (Pydantic). **[Verified, new]** Credential-like keys rejected (422). Evaluators built from the allowlist; retrieval metrics rejected as answer evaluators (400). |
| 2. Dataset version | `experiments/models.py:DatasetVersion` | Content fingerprint over all test cases (id, input, reference, metadata). |
| 3. Fingerprint | `ExperimentDefinition` | SHA-256 over dataset version + model config + evaluator configs. Recomputed and checked on every load (tamper/corruption check). |
| 4. Execute | `POST /experiments/{id}/execute` → `ExperimentManager.execute_experiment` | External adapters require `allow_external_inference`. **[Verified, new]** Evaluator configs must equal the definition's, and a RAG pipeline must match the declared retrieval configuration. |
| 5. Generation | `EvaluationRunner._run_single_test_case` → adapter `.generate` | An adapter exception is isolated to that case (`execution_error`); other cases continue. |
| 6. Retrieval (RAG) | `RagPipeline.generate` → `Retriever.retrieve` → `build_rag_prompt` | Ranked chunks with scores, the pipeline config and the exact prompt are stored in `provider_metadata["rag"]`. |
| 7. Evaluation | `_run_single_evaluator` | Each evaluator returns status ∈ {success, skipped, invalid_configuration, execution_error, output_validation_error}, plus `passed`, `score`, `label`, `explanation` and `evaluator_config`. Evaluator exceptions are isolated. |
| 8. Case verdict | `TestCaseResult.passed` | Policy: execution succeeded AND no evaluator said fail AND no evaluator errored AND at least one said pass. SKIPPED evaluators are ignored. |
| 9. Run aggregation | `RunResult` | `num_succeeded` (execution) and `num_passed` / `pass_rate` (evaluation) are separate. Run status (succeeded / partially / failed) is about execution only. |
| 10. Persistence | `LocalFileExperimentRepository.save_run` | Run stores a full config snapshot. **[Verified, new]** It also stores `adapter_id`. |
| 11. Post-hoc analysis | `api/services.py:DiagnosticsService` | Retrieval metrics (at the run's own K, **new**), consolidated diagnosis, lexical sentence evidence, robustness pairs. Computed on read and cached in memory, never persisted. |
| 12. Serialization | `to_run_detail` / `to_run_summary` | Secrets redacted for display. Adds `executed_model_ids` and `simulated_components` (**new**). |
| 13. UI | `frontend/src/pages/ResultsPage.tsx` | Per-run verdict bar (passed / wrong / not scored / crashed), cause list, per-question evidence (answers, checks, diagnosis, passages, sentence support, telemetry), "Do the checks agree?" (**new**), simulated-run notice (**new**). |

## D. Metrics and their exact semantics

| Metric | Definition | Denominator | Limits |
|---|---|---|---|
| Execution success | Adapter returned without exception | all cases | Says nothing about correctness. |
| Case pass rate (`pass_rate`) | `passed` cases / all cases (policy in C.8) | **all** cases, including crashed and unscored | One strict evaluator (e.g. exact match) drags it down. Read it with the per-evaluator rates and the 4-state breakdown. |
| `exact_match` | Equal after case and whitespace normalization only, against `reference_answer` | cases with a reference | Rejects correct paraphrases and extra words ("Paris." ≠ "Paris" only if punctuation differs; "The capital is Paris" fails). |
| `squad_match` (**new**) | SQuAD v1.1 normalization; pass = EM against any gold answer; score = max token F1 | cases with a reference | Lexical; a correct paraphrase fails, and a right span inside a wrong sentence can score high F1. |
| `semantic_similarity` | Similarity ≥ threshold (0.85) | cases with a reference | **In the API registry the backend is a fake (exact string equality).** BERTScore is available only with the optional `semantic` extra. It is now labelled "simulated" in the UI. |
| `llm_judge` (**now used**) | Judge picks correct / partially_correct / incorrect vs the reference; pass only for `correct` | cases with a reference | Judge bias, prompt sensitivity, self-preference when judge = generator. Not validated against humans. |
| `llm_faithfulness_judge` | Judge picks supported / unsupported / contradicted / undetermined vs retrieved context; `undetermined` → `passed = None` | RAG cases with a non-empty answer | In EXP-001 it rated the abstention "I don't know" as `supported`, which is a validity problem. The API registry version is a canned fake. |
| Hit@K | 1 if any relevant chunk is in the top K | RAG cases with ground truth | Ground truth is a labeling choice (see EXP-001 strict vs lenient). |
| Recall@K | \|relevant ∩ top K\| / \|relevant\| | same | Equals Hit@K when there is one relevant chunk. |
| MRR | 1 / rank of the first relevant chunk (0 if none); averaged across cases | same | Only over the retrieved list. |
| Evaluator agreement | Same verdict / compared cases; Cohen's kappa (**new**) | cases where both evaluators produced a verdict | Kappa is undefined when one evaluator is constant. |
| Latency | **[New]** Successful attempt only; total time including retries is kept in `total_elapsed_ms` | generated cases | Includes model loading on a cold call; the EXP-001 runner warms models up first. |

**Not implemented [Verified]:** Precision@K, context relevance (per-chunk relevance grading), answer relevance, NLI-based claim verification, and claim extraction (statements are split into sentences, not claims). Context utilisation exists only as a lexical word-overlap baseline (`rag/context_support.py`), which the code itself says must not be read as faithfulness.

## E. Reproducibility model

**[Verified]**

- **Fingerprints.** Dataset version and experiment config are SHA-256 hashes of canonical JSON. A run snapshots the dataset, model config, evaluator configs and fingerprint, and the fingerprint is re-verified on load.
- **Model provenance.** Provider responses record the reported model id, `system_fingerprint`, finish reason, attempts and token source (`provider_reported` / `unavailable`).
- **[New]** Runs record which adapter executed them. Judges record their full generation settings. The EXP-001 runner stores Ollama version, model digests, git commit and platform in experiment metadata.

**Gaps:**

- The *code version* is not recorded by the API path; only the study script records it.
- Random seeds are only recorded where passed in `extra_params`.
- Fixtures are regenerated with new random run ids each time (noisy diffs).

**Configuration reproducibility ≠ execution determinism.** EXP-001 measured determinism directly; see section G.

## F. Failure diagnosis methodology and limits

`diagnostics/diagnosis.py` applies an ordered rule list to the evidence already computed:

1. Execution error.
2. No evidence → `insufficient_evidence`.
3. Correctness evaluators disagree → `evaluator_disagreement`.
4. **[Changed]** Retrieval miss + wrong answer → `retrieval_failure`.
5. Faithfulness `contradicted` → `contradicted`.
6. `unsupported` → `unsupported`.
7. Retrieval hit + wrong → `generation_failure`.
8. Correct → `correct`.
9. Wrong without retrieval evidence → `incorrect`.
10. Otherwise → `undetermined`.

Every other signal is kept as a secondary finding.

Previously faithfulness came before retrieval, so a wrong answer after a retrieval miss was labelled "unsupported", a downstream symptom. The seed dataset itself labels that case `retrieval_failure`. Rules now attribute a failure to the **earliest failing pipeline stage**. This is a methodological choice the team can revert; it is tested in `tests/diagnostics/test_diagnosis.py`.

**Limits [Verified]:**

- Labels depend on which evaluators are configured. With exact match and an LLM judge together, every correct-but-verbose answer becomes `evaluator_disagreement`; EXP-001 shows this directly.
- Retrieval categories depend on ground-truth labels.
- A label is a rule outcome over observed evidence, not a causal explanation of model internals.

## G. Strengths and weaknesses

**Strengths [Verified]**

- Clean separation of execution vs evaluation, explicit evaluator statuses, and isolation of failures to a single case.
- Evidence is retained end to end: the prompt, retrieved chunks with scores, and evaluator config and explanation.
- Compatibility checks before comparing runs (different dataset versions are refused).
- Allowlisted execution, opt-in external inference, secrets redacted in responses.
- A large offline test suite: 468 backend tests, 50 frontend tests.

**Weaknesses found and fixed on this branch [Verified; each has a regression test]**

| # | Defect | Effect |
|---|---|---|
| 1 | Fake judge replies used `{verdict, explanation}`; the parser requires `{outcome, reasoning}` | Every faithfulness result was a validation error, so every RAG case "failed" (0/7) |
| 2 | `create_app(settings)` ignored settings in route dependencies | Each test run wrote junk experiments into `./experiments` (how the 31 junk files appeared) |
| 3 | Retrieval metrics accepted as answer evaluators | Execution persisted a run, then the API returned HTTP 500 |
| 4 | Runs recorded only the experiment's declared model | A mock run could be stored as "gpt-4o-mini"; benchmark runs were indistinguishable |
| 5 | The OpenAI adapter sent the experiment label as the model name | `openai-gpt-4o-mini` would have called a model named after the experiment |
| 6 | Same fingerprint but different adapters compared as "identical configuration … stochasticity" | Model comparisons were mislabelled as repeat trials |
| 7 | Retrieval scored at fixed K=1 | A top-1 vs top-2 comparison could never show a retrieval difference |
| 8 | A top-2 experiment could run on the top-1 pipeline (the UI default) | The run's recorded config did not produce it |
| 9 | Evaluators matched by name only at execution | A fake and a real judge could be swapped silently |
| 10 | Secrets in `extra_params` persisted to disk | Violated the no-secrets rule (redacted only in responses) |
| 11 | Latency included retry back-off sleeps | Biased RQ7 measurements |
| 12 | Diagnosis precedence (section F) | Downstream symptom labelled as the cause |
| 13 | Housekeeping | `openai` missing from dependencies; mypy not installed; 23 ruff errors; plain `pytest` / `npm test` failed; stray stores and a machine-specific script |

**Remaining weaknesses**

- **In the API, the only real evaluators are exact match and squad_match.** The judge and similarity entries in the registry are fakes (now labelled as such). Real judges are wired only in the study script, because judge calls need an opt-in and a model choice that the API does not yet expose.
- **Synchronous execution.** A real 60-question run takes minutes inside one HTTP request.
- **List endpoints parse every run file on each request.** That is fine at tens of runs and slow at thousands. SQLite would be the next step, not Postgres.
- **The case pass rate uses one strict any-fail policy** across evaluators measuring different things. The UI's 4-state breakdown and per-evaluator rates mitigate this, but the headline number is still dominated by the strictest evaluator.
- **Robustness covers only hand-labelled perturbation pairs inside a dataset.** There is no generator of perturbations.
- **Frontend fixtures are regenerated with random ids** (large diffs).

## H. Architectural and methodological risks

1. **Fake evaluators read as real.** This is now mitigated by `simulated_components` from the server, but any new fake must set the markers (`adapter: mock`, `backend: fake`, a `fake-` model id).
2. **Self-preference of LLM judges.** When the judge and the generator are the same model, model comparisons are confounded. Use a different judge family, or human labels on a sample.
3. **Judge validity is unmeasured.** Without even 50–100 human-labelled answers, "judge accuracy" is unknown.
4. **Small-n claims.** At n = 60, 95% intervals are ±10–13 points. Effects smaller than that cannot be distinguished from noise.
5. **Benchmark contamination.** SQuAD and other public sets are likely in pre-training data. Closed-book arms are required to interpret retrieval effects.
6. **Team integration.** The shared contract is `ExperimentRun` JSON plus API schemas. New fields were added as optional to keep old records loadable, but the fake-judge config changed, so old demo experiments must be reseeded (done here).

## I. Existing vs incomplete / unverified capabilities

| Capability | Status |
|---|---|
| Experiment definition, versioning, fingerprinting, atomic persistence | Working [Verified] |
| Offline mock and RAG demo execution, API and UI | Working [Verified in browser, live and fixture modes, zero console errors] |
| Real local models through the OpenAI-compatible adapter (Ollama) | Working [Verified, EXP-001] |
| Real OpenAI API (gpt-4o-mini, gpt-4o) | Code path tested offline with a fake client; **not run against the real API** (no key used) |
| Real LLM judges (correctness and faithfulness) | Working in the study script [Verified]; not available from the UI |
| Real retrieval embeddings | OpenAI-compatible embeddings [Verified, EXP-001]; SentenceTransformer path not run |
| BERTScore semantic similarity | Opt-in integration test skipped (dependency not installed) [Unverified] |
| Retrieval Hit/Recall/MRR, diagnosis, robustness pairs, comparison, trends, agreement + kappa | Working [Verified by tests and sweep] |
| Precision@K, context relevance, NLI faithfulness, claim extraction, human annotation, perturbation generation | Not implemented |
| Benchmarks (multi-adapter, repetitions) | Working [Verified]; synchronous |

### EXP-001 in one paragraph [Verified, n = 60 per arm]

Setup: local models (qwen3:4b, gemma3:1b) on a 60-question SQuAD subset with a 266-paragraph
corpus.

- **Retrieval:** top-3 retrieval raised LLM-judge correctness from 13% to 58% (qwen) and from 7% to
  50% (gemma). This is a large, highly significant paired effect.
- **Retrieval depth and model:** top-1 vs top-3, and qwen vs gemma, were *not* distinguishable at
  this sample size.
- **Evaluator choice:** the same 60 answers score 30% (exact match), 43% (SQuAD EM) or 58% (LLM
  judge). That spread is larger than any system difference measured.
- **Faithfulness judge:** it labels abstentions "supported".
- **Determinism:** temperature 0 with a fixed seed was not fully deterministic (57/60 identical).

Details: `research/experiments/EXP-001_local_rag_reliability/results.md`.

## J. Research questions that should guide the remaining work

Ordered by value for a defensible capstone:

1. **RQ3 evaluator reliability.** How well do exact match, SQuAD match and LLM judges agree with *human* labels? This is the cheapest high-value addition: label about 100 EXP-001 answers, then compute accuracy, kappa and confusion against humans.
2. **RQ2 retrieval effect with contamination control.** Closed book vs RAG, with retrieval quality measured by strict and lenient labels. EXP-001 gives a first answer.
3. **RQ6 failure attribution validity.** On human-labelled cases, how often does the rule-based category match a human's judgement of the cause?
4. **RQ4/RQ5 consistency and robustness.** Repeated sampling at T > 0 (EXP-001 has a first measurement), then paraphrased questions and context-order shuffles.
5. **RQ1 model comparison** with a judge that is not one of the compared models.

## K. Prioritised roadmap (about 4–5 months)

Each item names the RQ it serves. Months are relative to now (late Nov 2026).

| When | Work | Serves | Owner suggestion |
|---|---|---|---|
| Weeks 1–2 | Merge this branch after review; teammates reseed their local stores; adopt `simulated_components` markers for any new fake | integrity | Tech lead |
| Weeks 1–3 | **Human annotation set**: 100 EXP-001 answers × 2 annotators, guideline doc, inter-annotator kappa | RQ3, RQ6 | 2 members |
| Weeks 2–4 | Judge ablation: second judge family (e.g. llama3.2:3b or gpt-4o-mini if budget allows), rubric wording variants, judge temperature | RQ3 | 1 member |
| Month 2 | API: real judges in the registry behind `allow_external_inference`, configurable judge model; async execution (background task + polling) | usability, RQ1 | backend pair |
| Month 2 | Perturbation generator (question paraphrase, context-order shuffle, distractor injection) using the existing `perturbation_of` metadata | RQ5 | 1 member |
| Month 2–3 | EXP-002 (judge validity vs humans), EXP-003 (robustness), EXP-004 (model comparison with a neutral judge) | RQ1–RQ5 | team |
| Month 3 | Precision@K and graded context relevance (judge or human) for retrieval | RQ2 | 1 member |
| Month 3–4 | SQLite repository behind the existing `ExperimentRepository` interface, only if run counts make JSON listing slow | scale | backend |
| Month 4 | Frontend: per-evaluator rates with intervals, human-vs-judge view, experiment comparison by arm | communication | Sai |
| Month 4–5 | Report and defence: related work (RAGAS, ARES, TruLens, HELM, LLM-as-judge literature), methods, results with limitations | all | team |

**What not to do [Recommendation]:**

- No Postgres, Redis or Kubernetes.
- Don't make LangChain or LlamaIndex a core dependency.
- No new dashboards without a research question behind them.
- Don't add evaluators faster than you can validate them.
