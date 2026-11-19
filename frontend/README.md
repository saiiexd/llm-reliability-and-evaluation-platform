# Research console (frontend)

A web UI for the LLM Reliability platform. It answers one question:
**does a language model answer a fixed set of questions correctly, and when it
doesn't, why?**

## The idea in six words

| Word | Meaning | Example |
| --- | --- | --- |
| **Experiment** | A fixed set of questions + the model to test + the checks to run. Defined once. | "Reliability console demonstration": 7 questions, `mock-adapter-v1`, exact match + LLM judge |
| **Run** | One execution of an experiment: the model answers every question and every answer is checked. An experiment can have many runs. | Run `418fb8f6`, 2 Nov 02:14 |
| **Question** (test case) | One input with its expected answer. | "What is the capital of France?" → "Paris" |
| **Check** (evaluator) | One way of judging an answer. Each check runs independently. | *Exact match*: text must equal "Paris". *Faithfulness (LLM judge)*: a second model checks the answer is backed by the retrieved passages. |
| **Verdict** | The outcome for one answer, combining all its checks (see below). | Passed / Wrong / Not scored / Crashed |
| **Diagnosis** | The backend's best guess at *why* an answer failed. | Retrieval failure: the search step never found the passage with the answer |

So the data flows: **Experiment → Run → one result per Question → one result per Check → Verdict + Diagnosis.**

### The four verdicts

Defined in `src/lib/derived.ts` → `testCaseVerdict()`.

| Verdict | Rule |
| --- | --- |
| **Passed** | Every check ran and accepted the answer. |
| **Wrong** | At least one check ran and rejected the answer. |
| **Not scored** | No check rejected it, but at least one check couldn't run (e.g. the LLM judge returned invalid output). |
| **Crashed** | The model call itself failed, so there was no answer to check. |

Why "Not scored" exists: the backend's `passed` flag is `false` both when an
answer is wrong *and* when a check merely errored. Showing both as "Failed"
made the correct answer "Paris" look wrong. The set of **Passed** answers is
identical to the backend's; only "not passed" is split into Wrong vs Not scored.

### RAG terms (Retrieval tab)

RAG (retrieval-augmented generation) = the model first **searches** a document
collection for relevant passages, then **answers** using them. So a wrong answer
has two possible causes:

- **Retrieval failure**: search didn't find the passage containing the answer.
- **Generation failure**: search found it, but the model still answered wrongly.

Metrics: *hit@k* = was the right passage in the top k results (0 or 1, averaged);
*MRR* (mean reciprocal rank) = 1 if the right passage ranked 1st, ½ if 2nd, ⅓ if 3rd…, averaged.

**Consistency**: ask the same question reworded; a reliable model gets the same verdict both times.

## Run it

From the repo root:

```fish
.venv/bin/python run.py          # starts backend :8000 and frontend :5173
```

Open http://localhost:5173. Frontend only: `cd frontend; npm run dev`.

Tests (Node 26 needs the flag; Node's own `localStorage` hides jsdom's):

```fish
cd frontend
env NODE_OPTIONS=--no-experimental-webstorage npm test
npm run build      # typecheck + production build
npm run lint
```

## Use it (5-minute tour)

1. **Overview**: overall pass rate, the latest run, recent runs. Click a run.
2. **Results**: the summary sentence and bar tell you the run in one line.
   Filter questions by verdict. Click a question to see expected vs. actual
   answer, what each check said (including the error if a check couldn't run),
   and the diagnosis. Retrieved passages and token usage are folded at the bottom.
3. **Experiments → an experiment**: run it again with a chosen model (*Run it*),
   or run several models / repeats at once (*Compare models*). Past runs are listed.
   *New experiment* creates one: pick questions, a model and checks.
4. **Analysis**: one question per tab.
   - *Compare runs*: is the comparison fair, what changed, and did results change.
   - *Why answers fail*: all failures ranked by cause; click a cause to list them.
   - *Retrieval*, *Consistency*, *Speed & cost*: as named.

**Live API vs Demo data** (top right): *Live* talks to the backend. *Demo* uses
bundled sample data from `src/data/fixtures/` and shows a notice on every page.
If the backend is down in Live mode, the app says so. It never silently
switches to demo data.

## How the code fits together

```
page (src/pages/*.tsx)
  └─ hook: useRuns(), useRun(id), useExperiment(id) …   src/data/useData.ts
       └─ useAsyncData(liveFn, fixtureFn)  picks one based on Live/Demo mode
            ├─ live:  ApiClient → fetch → FastAPI /api/v1/…   src/api/client.ts
            └─ demo:  bundled JSON                             src/data/dataAccess.ts
  └─ pure helpers: verdicts, labels, formatting                src/lib/derived.ts
  └─ shared UI: PageHeader, VerdictBar, Tag, DataStateView     src/components/
```

| Folder | What lives there |
| --- | --- |
| `pages/` | One file per screen. `AnalysisPage` is the tab shell; each tab is its own page file. |
| `components/` | Reusable pieces. `Verdict.tsx` (verdict tag/bar/sentence), `DataStateView` (loading + error), `NewExperimentForm`. |
| `data/` | Hooks and the live/demo switch. Pages never call `fetch` directly. |
| `api/client.ts` | Typed HTTP client for every backend endpoint. |
| `context/ConnectionContext.tsx` | Live/Demo mode, health check, and the run/benchmark actions. |
| `lib/derived.ts` | Logic with no UI: verdicts, pass counts, plain-language labels. Most unit-tested. |
| `types/domain.ts` | TypeScript shapes of backend data. |
| `styles/index.css` | Design tokens. Change colours here, not in components. |

### Things that will trip you up

- **List vs detail shapes.** `GET /runs` returns *summaries* (`num_passed`,
  `num_test_cases`, no per-question results). `GET /runs/{id}` returns the full
  run. Same for experiments (`model_id` flat in the list, `model_config.model_id`
  nested in detail). Use `runPassCounts()` / `experimentModel()` which handle both.
- **Every non-pass isn't "wrong".** Use `testCaseVerdict()`, not `result.passed`.
- **Design rule.** Greys for structure; the single accent colour (`accent`) only
  for wrong answers and errors; every state also has a text label.

## Known issues (backend, as of 2026-11-19)

- The LLM faithfulness judge errors on every answer
  (`Judge output 'outcome' must be one of …, got None`). Until fixed, most
  answers are *Not scored* and the overall pass rate looks very low.
- `tests/api/test_benchmarks.py` and `test_providers.py` write junk
  experiments into the real `./experiments` folder (`create_app(settings)`
  doesn't override the `get_settings` dependency). Don't run pytest against
  data you care about until this is fixed.
- The *Why answers fail* and *Retrieval* tabs fetch every run one by one;
  fine for tens of runs, slow for hundreds.
- `useData.ts` still exports hooks no page uses (analytics, trends, …); they
  map backend endpoints and can be used or removed later.
