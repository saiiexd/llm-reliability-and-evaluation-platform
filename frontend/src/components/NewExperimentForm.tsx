/**
 * Form for creating an experiment: pick questions, a model, and the checks.
 * Moved out of ExperimentsPage so the page itself stays a simple list.
 */
import { useState, type ReactNode } from "react";
import { useConnection } from "../context/ConnectionContext";

const PRESET_DATASETS = [
  {
    name: "qa-geography-baseline-v1",
    title: "General knowledge",
    description: "Short factual questions (capitals, elements, dates) with one correct answer each.",
    testCases: [
      { id: "tc-1", input: "What is the capital of France?", reference_answer: "Paris", metadata: {} },
      { id: "tc-2", input: "Bananas are rich in which mineral?", reference_answer: "Potassium", metadata: {} },
      { id: "tc-3", input: "What is the largest ocean on Earth?", reference_answer: "Pacific Ocean", metadata: {} },
      { id: "tc-4", input: "Which element has atomic number 1?", reference_answer: "Hydrogen", metadata: {} },
      { id: "tc-5", input: "What year did the Apollo 11 mission land on the Moon?", reference_answer: "1969", metadata: {} },
    ],
  },
  {
    name: "rag-groundedness-evaluation-v1",
    title: "Answers from documents",
    description: "Questions the model must answer using passages it looks up first (RAG).",
    testCases: [
      { id: "rag-1", input: "What is the capital of France?", reference_answer: "Paris", metadata: {} },
      { id: "rag-2", input: "Which fruit is high in potassium?", reference_answer: "Banana", metadata: {} },
      { id: "rag-3", input: "Who wrote Romeo and Juliet?", reference_answer: "William Shakespeare", metadata: {} },
    ],
  },
];

const SQUAD_PRESET = "squad-v1.1-dev-subset-v1";

const AVAILABLE_MODELS = [
  { id: "mock-adapter-v1", name: "Mock model (simulated)", desc: "Fixed, offline answers for the demo questions. Exercises the platform; its scores say nothing about real models." },
  { id: "rag-pipeline-v1", name: "Demo RAG pipeline (simulated)", desc: "Looks up passages in a two-document toy corpus with fake embeddings, then gives canned answers." },
  { id: "echo-adapter", name: "Echo (simulated)", desc: "Repeats the question back. A baseline that should fail every check." },
  { id: "local", name: "Local or self-hosted model", desc: "A real model on an OpenAI-compatible server (Ollama, vLLM, LM Studio). Type its served name below." },
  { id: "openai-gpt-4o-mini", name: "OpenAI GPT-4o mini (hosted)", desc: "A real hosted model. Costs money; needs OPENAI_API_KEY on the backend." },
];

const AVAILABLE_EVALUATORS = [
  { id: "exact_match", name: "Exact match", desc: "The answer must equal the expected answer (ignoring case and spacing). Strict." },
  { id: "squad_match", name: "SQuAD match", desc: "Official SQuAD normalisation (articles, punctuation) against every accepted answer; score is token F1." },
  { id: "semantic_similarity", name: "Similar meaning (simulated)", desc: "Credit for paraphrases. The API uses a fake similarity backend, so treat its scores as a demo." },
  { id: "llm_judge", name: "Correctness (LLM judge)", desc: "A judge model compares the answer with the expected answer. Needs a judge model below." },
  { id: "llm_faithfulness_judge", name: "Faithfulness (LLM judge)", desc: "A judge model checks the answer is backed by the retrieved passages. Without a judge model it is a canned demo judge." },
];

const inputClass =
  "w-full rounded-md border border-line bg-paper px-3 py-2 text-sm text-ink placeholder:text-ink-soft focus:border-ink focus:outline-none";

function Choice({
  selected,
  onClick,
  title,
  children,
}: {
  selected: boolean;
  onClick: () => void;
  title: string;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={selected}
      className={`rounded-lg border p-4 text-left transition-colors ${
        selected ? "border-ink bg-paper-raised" : "border-line hover:bg-paper-dim"
      }`}
    >
      <p className="text-sm font-medium">{title}</p>
      <p className="mt-1 text-xs leading-relaxed text-ink-soft">{children}</p>
    </button>
  );
}

function Step({ n, title, hint, children }: { n: number; title: string; hint: string; children: ReactNode }) {
  return (
    <fieldset className="space-y-3 border-t border-line pt-6">
      <legend className="sr-only">{title}</legend>
      <div>
        <p className="text-xs text-ink-soft">Step {n}</p>
        <p className="font-medium">{title}</p>
        <p className="text-sm text-ink-soft">{hint}</p>
      </div>
      {children}
    </fieldset>
  );
}

export function NewExperimentForm({
  onCreated,
  onCancel,
}: {
  onCreated: (experimentId: string) => void;
  onCancel: () => void;
}) {
  const { dataMode, client } = useConnection();
  const [expName, setExpName] = useState("");
  const [expId, setExpId] = useState("");
  // The ID follows the name until the user edits the ID themselves.
  const [idEdited, setIdEdited] = useState(false);
  const [expDesc, setExpDesc] = useState("");
  const [datasetIndex, setDatasetIndex] = useState(0);
  const [targetModel, setTargetModel] = useState("mock-adapter-v1");
  const [localModelId, setLocalModelId] = useState("qwen3:4b-instruct");
  const [retrieval, setRetrieval] = useState<"none" | "topk">("topk");
  const [topK, setTopK] = useState(3);
  const [questionLimit, setQuestionLimit] = useState(60);
  const [judgeModelId, setJudgeModelId] = useState("");
  const [evaluators, setEvaluators] = useState<string[]>(["exact_match"]);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [temperature, setTemperature] = useState(0.0);
  const [maxTokens, setMaxTokens] = useState(100);
  const [seed, setSeed] = useState("42");
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const usingSquad = datasetIndex === PRESET_DATASETS.length;
  const usesJudge = evaluators.includes("llm_judge") || evaluators.includes("llm_faithfulness_judge");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!expName.trim() || !expId.trim()) {
      setError("Give the experiment a name and an ID.");
      return;
    }
    if (evaluators.length === 0) {
      setError("Pick at least one check.");
      return;
    }
    if (evaluators.includes("llm_judge") && !judgeModelId.trim()) {
      setError("The correctness judge needs a judge model (for example qwen3:4b-instruct).");
      return;
    }
    if (targetModel === "local" && !localModelId.trim()) {
      setError("Type the served name of the local model.");
      return;
    }
    setCreating(true);
    setError(null);
    const id = expId.trim();
    const modelConfig = {
      model_id: targetModel === "local" ? localModelId.trim() : targetModel,
      temperature: showAdvanced ? temperature : 0.0,
      max_tokens: showAdvanced ? maxTokens : usingSquad ? 64 : 100,
      extra_params: showAdvanced && seed.trim() ? { seed: parseInt(seed, 10) } : {},
    };
    const judge = judgeModelId.trim() || undefined;
    try {
      if (dataMode === "live") {
        if (usingSquad) {
          await client.createExperiment({
            experiment_id: id,
            name: expName.trim(),
            description: expDesc.trim() || undefined,
            dataset_preset: SQUAD_PRESET,
            dataset_limit: questionLimit < 60 ? questionLimit : undefined,
            retrieval_top_k: retrieval === "topk" ? topK : undefined,
            judge_model_id: judge,
            model_config: modelConfig,
            evaluator_names: evaluators,
            metadata: {},
          });
        } else {
          const dataset = PRESET_DATASETS[datasetIndex];
          await client.createExperiment({
            experiment_id: id,
            name: expName.trim(),
            description: expDesc.trim() || undefined,
            dataset_name: dataset.name,
            test_cases: dataset.testCases,
            judge_model_id: judge,
            model_config: modelConfig,
            evaluator_names: evaluators,
            metadata: {},
          });
        }
      }
      onCreated(id);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setCreating(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6 rounded-xl border border-line bg-paper-dim p-6">
      <div>
        <h2 className="text-lg font-semibold">New experiment</h2>
        <p className="text-sm text-ink-soft">
          Three choices. Everything you pick is saved with the experiment and fingerprinted; you can run it right
          after saving.
        </p>
      </div>

      {error && (
        <p className="rounded-lg border border-accent/40 bg-accent-soft/40 px-3 py-2 text-sm text-ink">{error}</p>
      )}

      <div className="grid gap-4 sm:grid-cols-2">
        <label className="block text-sm">
          <span className="mb-1 block text-ink-soft">Name</span>
          <input
            id="exp-name"
            required
            placeholder="e.g. Geography baseline"
            value={expName}
            onChange={(e) => {
              setExpName(e.target.value);
              if (!idEdited) {
                setExpId(e.target.value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, ""));
              }
            }}
            className={inputClass}
          />
        </label>
        <label className="block text-sm">
          <span className="mb-1 block text-ink-soft">ID (used in the URL)</span>
          <input
            id="exp-id"
            required
            placeholder="e.g. geography-baseline"
            value={expId}
            onChange={(e) => {
              setIdEdited(true);
              setExpId(e.target.value);
            }}
            className={`${inputClass} font-mono`}
          />
        </label>
        <label className="block text-sm sm:col-span-2">
          <span className="mb-1 block text-ink-soft">What are you trying to find out? (optional)</span>
          <input
            id="exp-desc"
            placeholder="e.g. Does the mock model get capitals right?"
            value={expDesc}
            onChange={(e) => setExpDesc(e.target.value)}
            className={inputClass}
          />
        </label>
      </div>

      <Step n={1} title="Questions" hint="The questions to ask, each with its expected answer.">
        <div className="grid gap-3 sm:grid-cols-2">
          {PRESET_DATASETS.map((ds, i) => (
            <Choice
              key={ds.name}
              selected={datasetIndex === i}
              onClick={() => setDatasetIndex(i)}
              title={`${ds.title} · ${ds.testCases.length} questions`}
            >
              {ds.description}
            </Choice>
          ))}
          <Choice
            selected={usingSquad}
            onClick={() => {
              setDatasetIndex(PRESET_DATASETS.length);
              if (targetModel !== "local" && !targetModel.startsWith("openai")) setTargetModel("local");
              setEvaluators((prev) => (prev.includes("squad_match") ? prev : [...prev, "squad_match"]));
            }}
            title="SQuAD v1.1 subset · 60 questions"
          >
            Real reading-comprehension questions over 266 Wikipedia paragraphs, with passage labels so
            retrieval can be scored. CC BY-SA 4.0 (Rajpurkar et al., 2016). The dataset EXP-001 used.
          </Choice>
        </div>
        {usingSquad && (
          <div className="grid gap-4 rounded-lg border border-line p-4 sm:grid-cols-2">
            <label className="block text-sm">
              <span className="mb-1 block text-ink-soft">Use the first N questions</span>
              <input
                id="question-limit"
                type="number"
                min={1}
                max={60}
                value={questionLimit}
                onChange={(e) => setQuestionLimit(Math.max(1, Math.min(60, parseInt(e.target.value) || 60)))}
                className={inputClass}
              />
              <span className="text-xs text-ink-soft">10 is enough to try it; 60 for a real comparison.</span>
            </label>
            <div className="text-sm">
              <span className="mb-1 block text-ink-soft">Retrieval</span>
              <div className="flex gap-2">
                <label className="flex items-center gap-1.5">
                  <input type="radio" checked={retrieval === "none"} onChange={() => setRetrieval("none")} className="accent-ink" />
                  None (closed book)
                </label>
                <label className="flex items-center gap-1.5">
                  <input type="radio" checked={retrieval === "topk"} onChange={() => setRetrieval("topk")} className="accent-ink" />
                  Top
                  <input
                    id="top-k-input"
                    type="number"
                    min={1}
                    max={20}
                    value={topK}
                    disabled={retrieval !== "topk"}
                    onChange={(e) => setTopK(Math.max(1, Math.min(20, parseInt(e.target.value) || 3)))}
                    className="w-14 rounded-md border border-line bg-paper px-2 py-1"
                  />
                  passages
                </label>
              </div>
              <span className="mt-1 block text-xs text-ink-soft">
                Retrieval uses nomic-embed-text on the same server; run it with the "+ SQuAD passage retrieval" model.
              </span>
            </div>
          </div>
        )}
      </Step>

      <Step n={2} title="Model" hint="The model that will answer the questions.">
        <div className="grid gap-3 sm:grid-cols-2">
          {AVAILABLE_MODELS.map((m) => (
            <Choice key={m.id} selected={targetModel === m.id} onClick={() => setTargetModel(m.id)} title={m.name}>
              {m.desc}
            </Choice>
          ))}
        </div>
        {targetModel === "local" && (
          <label className="block max-w-md text-sm">
            <span className="mb-1 block text-ink-soft">Served model name</span>
            <input
              id="local-model-id"
              value={localModelId}
              onChange={(e) => setLocalModelId(e.target.value)}
              placeholder="e.g. qwen3:4b-instruct"
              className={`${inputClass} font-mono`}
            />
          </label>
        )}
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="text-sm text-ink-soft hover:text-ink"
        >
          {showAdvanced ? "Hide" : "Show"} advanced settings
        </button>
        {showAdvanced && (
          <div className="grid gap-4 rounded-lg border border-line p-4 sm:grid-cols-2">
            <label className="block text-sm">
              <span className="mb-1 block text-ink-soft">Temperature: {temperature}</span>
              <input
                id="temp-input"
                type="range"
                min={0}
                max={1}
                step={0.1}
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="w-full accent-ink"
              />
              <span className="text-xs text-ink-soft">0 gives the same answer every time; higher varies more.</span>
            </label>
            <label className="block text-sm">
              <span className="mb-1 block text-ink-soft">Max answer length (tokens)</span>
              <input
                id="tokens-input"
                type="number"
                min={10}
                max={1000}
                value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value) || 100)}
                className={inputClass}
              />
            </label>
            <label className="block text-sm">
              <span className="mb-1 block text-ink-soft">Seed (blank for none)</span>
              <input
                id="seed-input"
                value={seed}
                onChange={(e) => setSeed(e.target.value.replace(/[^0-9]/g, ""))}
                className={`${inputClass} font-mono`}
              />
              <span className="text-xs text-ink-soft">
                Sent to the server; with temperature above 0, leave it blank to measure run-to-run variation.
              </span>
            </label>
          </div>
        )}
      </Step>

      <Step n={3} title="Checks" hint="How each answer is judged. Each check runs separately.">
        <div className="grid gap-3 sm:grid-cols-3">
          {AVAILABLE_EVALUATORS.map((ev) => (
            <Choice
              key={ev.id}
              selected={evaluators.includes(ev.id)}
              onClick={() =>
                setEvaluators((prev) => (prev.includes(ev.id) ? prev.filter((x) => x !== ev.id) : [...prev, ev.id]))
              }
              title={ev.name}
            >
              {ev.desc}
            </Choice>
          ))}
        </div>
        {usesJudge && (
          <label className="block max-w-md text-sm">
            <span className="mb-1 block text-ink-soft">Judge model (served by the same OpenAI-compatible server)</span>
            <input
              id="judge-model-id"
              value={judgeModelId}
              onChange={(e) => setJudgeModelId(e.target.value)}
              placeholder="e.g. qwen3:4b-instruct"
              className={`${inputClass} font-mono`}
            />
            <span className="text-xs text-ink-soft">
              A judge that is also the answering model may favour its own answers; a different judge is safer.
            </span>
          </label>
        )}
      </Step>

      <div className="flex justify-end gap-2 border-t border-line pt-5">
        <button type="button" onClick={onCancel} className="rounded-md border border-line px-4 py-2 text-sm hover:bg-paper">
          Cancel
        </button>
        <button
          type="submit"
          disabled={creating}
          className="rounded-md bg-ink px-5 py-2 text-sm font-medium text-paper hover:opacity-90 disabled:opacity-50"
        >
          {creating ? "Saving…" : "Save experiment"}
        </button>
      </div>
    </form>
  );
}
