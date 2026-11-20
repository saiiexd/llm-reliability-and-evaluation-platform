# LLM Reliability & Evaluation Platform

A research platform for measuring **how and why LLM and RAG applications fail**. It runs a
versioned set of questions through a model or RAG pipeline, scores every answer with several
independent evaluators, keeps all the evidence (prompt, retrieved passages, output, each
evaluator's verdict and reasoning, configuration, provenance), attributes failures to a pipeline
stage, and compares runs only when the comparison is valid.

It is not a chatbot. The pipeline it studies is:

```
Question -> Retrieval -> Retrieved context -> Generation -> Answer -> Evaluation -> Diagnosis
```

## Documentation

| Document | What it is |
|---|---|
| [`docs/PROJECT_TECHNICAL_UNDERSTANDING.md`](docs/PROJECT_TECHNICAL_UNDERSTANDING.md) ([PDF](docs/PROJECT_TECHNICAL_UNDERSTANDING.pdf)) | Full technical report: research foundations, architecture and lifecycle diagrams, metric definitions, failure taxonomy, experimental design, data model, implementation audit, related work, risks, roadmap |
| [`research/experiments/EXP-001_local_rag_reliability/`](research/experiments/EXP-001_local_rag_reliability/) | First real experiment: definition (written before running), results, failure analysis, per-case data, archived runs |
| [`frontend/README.md`](frontend/README.md) | Console concepts, 5-minute tour, frontend code structure |
| [`research/project_definition.md`](research/project_definition.md), [`research/notes/`](research/notes/) | Project definition and design notes per component |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Engineering rules |

## Status (verified 2026-11-19)

- Backend: 473 tests passed, 3 skipped (opt-in integration tests); ruff and mypy clean.
- Frontend: 52 tests passed; ESLint clean; production build succeeds.
- Real local models (Ollama) verified end to end; EXP-001 ran 540 real generations.
- The demo experiments that ship with the repository use a mock model, fake embeddings and a
  canned judge. They exercise the platform and are marked **Simulated** in the console; do not
  read their numbers as model behaviour.

## Quick start

Prerequisites: Python 3.13, Node.js 20+ (verified on 26). Optional: [Ollama](https://ollama.com)
for real local models.

```bash
# Backend
python -m venv .venv && . .venv/bin/activate     # or: uv venv
pip install -e ".[dev]"                           # or: uv pip install -e ".[dev]"
pytest                                            # 473 passed, 3 skipped
ruff check src tests scripts && mypy

# Demo data (simulated) for the console
python scripts/seed_live_experiments.py

# Frontend
cd frontend && npm ci && npm test && npm run build && cd ..

# Run both servers (backend :8000, console :5173)
python run.py
```

- Console: http://localhost:5173 · API docs: http://127.0.0.1:8000/api/v1/docs
- Run separately: `python -m uvicorn llm_reliability.api.app:app --port 8000` and
  `cd frontend && npm run dev`.
- Tests never touch `./experiments`; each API test uses a temporary store.

## Running real models (local, no API key)

```bash
ollama pull qwen3:4b-instruct && ollama pull gemma3:1b && ollama pull nomic-embed-text
python scripts/run_reliability_study.py        # EXP-001 (~25 min on an RTX 3050 6 GB)
python scripts/analyze_reliability_study.py    # results.json, results_tables.md, per_case.csv
```

To see the archived EXP-001 runs without re-running:
`tar xzf research/experiments/EXP-001_local_rag_reliability/runs.tar.gz -C experiments`.

### Real models from the console

1. Start the backend with the server URL set:
   `LLM_RELIABILITY_OPENAI_BASE_URL=http://localhost:11434/v1 python run.py`.
2. **Experiments → New experiment.** Choose:
   - **SQuAD v1.1 subset**, with the first N questions, and closed book or top-K retrieval;
   - **Local or self-hosted model**, with its served name (e.g. `qwen3:4b-instruct`);
   - the checks. Give a judge model name to make the LLM judges real, and pick rubric v1 or v2.
3. **Run.** The matching adapter is pre-selected:
   - `openai-custom` for closed book;
   - `openai-custom-rag` for real retrieval over the SQuAD passages with `nomic-embed-text`.

   Tick the opt-in: every part that calls a model server is listed, including judges, and without
   the opt-in the server returns 400. The run executes in the background and shows its progress.
4. The run is **not** labelled Simulated. Its reproducibility record shows the git commit and
   library versions.

Hosted OpenAI adapters need `OPENAI_API_KEY` in the server environment (not tested: no key was
used). The same flow is available over the API: `POST /experiments` with `dataset_preset`,
`retrieval_top_k`, `judge_model_id` and `judge_rubric_version`, then `POST /experiments/{id}/jobs`
and poll `GET /jobs/{job_id}`.

Optional real BERTScore and SentenceTransformer embeddings, CPU only:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[semantic,embeddings]"
LLM_RELIABILITY_RUN_INTEGRATION_TESTS=1 pytest tests/evaluation/test_semantic_similarity_integration.py tests/rag/test_embeddings_integration.py
```

## How it works (one paragraph)

An **experiment definition** pins a dataset version, model configuration and evaluator
configurations under a SHA-256 fingerprint. **Executing** it through an allowlisted **adapter**
(mock, RAG pipeline, OpenAI or OpenAI-compatible, replay) is refused if the evaluators, RAG
settings or prompt template differ from the definition. Each test case is generated, then scored by every evaluator in
isolation; execution success and evaluation verdicts are kept separate. A run is persisted as one
JSON file with a full configuration snapshot and provenance (`adapter_id`,
`executed_model_ids`, `simulated_components`, and an `environment` record: git commit, Python and
library versions). Retrieval metrics, diagnoses and analytics are
computed when results are read. Details and diagrams: the technical report, sections 4–9.

## Evaluators and metrics

| Name | Measures | Notes |
|---|---|---|
| `exact_match` | Text equals the reference (case/whitespace normalized) | Strict; rejects paraphrases |
| `squad_match` | SQuAD-normalized match against any gold answer; score = token F1 | Lexical |
| `semantic_similarity` | BERTScore F1 ≥ threshold | The API registry uses a fake backend (labelled simulated). Real BERTScore (`.[semantic]`) is verified by opt-in integration tests |
| `llm_judge` | Correctness vs reference by an LLM rubric | Real judge when `judge_model_id` is set. Rubric v1, or v2 (all accepted answers) |
| `llm_faithfulness_judge` | Support by retrieved context | Real judge when `judge_model_id` is set; otherwise a canned demo judge (labelled simulated). v2 adds `no_claim` for abstentions |
| Hit@K, Recall@K, Precision@K, MRR | Retrieval vs `relevant_chunk_ids` | Computed automatically for RAG runs at the run's own K |
| Repeat consistency | Identical outputs and stable verdicts across repeats of one configuration and adapter | `/analytics/consistency`; Analysis → Reliability |

## API (all under `/api/v1`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health`, `/providers` | Service status; provider readiness (no secrets) |
| GET, POST | `/experiments` | List / create experiment definitions |
| GET | `/experiments/{id}`, `/experiments/{id}/runs` | Definition; its runs |
| POST | `/experiments/{id}/execute` | Run once with an allowlisted adapter (waits for the result) |
| POST | `/experiments/{id}/jobs`; GET `/jobs`, `/jobs/{job_id}` | Run in the background, with progress |
| GET | `/datasets` | Bundled dataset presets (SQuAD v1.1 subset) |
| POST | `/benchmarks/execute`; GET `/benchmarks/{id}/runs` | Several adapters × repetitions |
| GET | `/runs`, `/runs/{id}`, `/runs/{id}/test-cases[/{tc}]`, `/runs/{id}/summaries` | Runs, per-question results, summaries |
| GET | `/diagnostics[/{run}/{tc}]` | Failure diagnoses with evidence |
| GET | `/rag/retrieval-results[/{run}/{tc}]` | Retrieved passages and retrieval metrics |
| GET | `/robustness[/{run}]` | Baseline/perturbation comparisons |
| GET | `/analytics/overview`, `/evaluators`, `/failures`, `/retrieval`, `/robustness`, `/performance`, `/comparisons`, `/consistency`, `/trends/{experiment}` | Aggregates, agreement (Cohen's κ), comparisons, repeat consistency |

## Repository layout

```
src/llm_reliability/
  evaluation/    domain models, EvaluationRunner, evaluators, adapters, closed-book prompting, replay
  datasets/      loaders for bundled benchmark datasets (SQuAD subset, frozen paraphrases)
  rag/           chunking, embeddings, retriever, RagPipeline, retrieval evaluators, faithfulness judge
  experiments/   definitions, runs, fingerprints, ExperimentManager, JSON repository
  providers/     OpenAI / OpenAI-compatible adapters with retries and token provenance
  diagnostics/   failure taxonomy, diagnosis rules, sentence evidence, robustness pairs
  reliability/   per-evaluator summaries, reliability findings
  analytics/     aggregations, comparison + compatibility, agreement (kappa), repeat consistency,
                 human-annotation export and agreement, proportions (Wilson, McNemar)
  api/           FastAPI app, routers, schemas, allowlisted registries, live models, jobs
frontend/        React + TypeScript console (Overview, Experiments, Results, Analysis)
tests/           backend tests (pytest)
scripts/         demo seeding, fixture export, examples, dataset builders, study runners + analyses
data/            small versioned datasets (SQuAD v1.1 subset, methodology eval set)
experiments/     experiment store: definitions are versioned; runs/ is git-ignored
research/        project definition, design notes, experiments (EXP-001..003), annotation kit
docs/            technical report (Markdown + PDF)
```

## Data and secrets

- Experiment definitions are written to disk; requests containing credential-like keys
  (`api_key`, `secret_key`, `token`, `password`, ...) are rejected. Keys belong in server
  environment variables.
- `experiments/runs/` is git-ignored. Findings belong in `research/experiments/` as documents;
  EXP-001's raw runs are archived there as `runs.tar.gz`.
- SQuAD-derived data is CC BY-SA 4.0 (Rajpurkar et al., 2016).
