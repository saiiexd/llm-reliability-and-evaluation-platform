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

const AVAILABLE_MODELS = [
  { id: "mock-adapter-v1", name: "Mock model", desc: "Fixed, offline answers. No API key needed. Good for trying things out." },
  { id: "rag-pipeline-v1", name: "RAG pipeline", desc: "Looks up passages in a local corpus, then answers from them." },
  { id: "echo-adapter", name: "Echo", desc: "Repeats the question back. Useful as a baseline that should fail." },
  { id: "openai-gpt-4o-mini", name: "OpenAI GPT-4o mini", desc: "A real hosted model. Needs an OpenAI API key on the backend." },
];

const AVAILABLE_EVALUATORS = [
  { id: "exact_match", name: "Exact match", desc: "The answer must match the expected answer (ignoring case and spacing)." },
  { id: "semantic_similarity", name: "Similar meaning", desc: "Gives credit for paraphrases, using text embeddings." },
  { id: "llm_faithfulness_judge", name: "Faithfulness (LLM judge)", desc: "A second model checks the answer is backed by the looked-up passages." },
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
  const [expDesc, setExpDesc] = useState("");
  const [datasetIndex, setDatasetIndex] = useState(0);
  const [targetModel, setTargetModel] = useState("mock-adapter-v1");
  const [evaluators, setEvaluators] = useState<string[]>(["exact_match"]);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [temperature, setTemperature] = useState(0.0);
  const [maxTokens, setMaxTokens] = useState(100);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
    setCreating(true);
    setError(null);
    const dataset = PRESET_DATASETS[datasetIndex];
    const id = expId.trim();
    try {
      if (dataMode === "live") {
        await client.createExperiment({
          experiment_id: id,
          name: expName.trim(),
          description: expDesc.trim() || undefined,
          dataset_name: dataset.name,
          test_cases: dataset.testCases,
          model_config: {
            model_id: targetModel,
            temperature: showAdvanced ? temperature : 0.0,
            max_tokens: showAdvanced ? maxTokens : 100,
          },
          evaluator_names: evaluators,
          metadata: {},
        });
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
        <p className="text-sm text-ink-soft">Three choices. You can run it right after saving.</p>
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
              if (!expId) {
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
            onChange={(e) => setExpId(e.target.value)}
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
        </div>
      </Step>

      <Step n={2} title="Model" hint="The model that will answer the questions.">
        <div className="grid gap-3 sm:grid-cols-2">
          {AVAILABLE_MODELS.map((m) => (
            <Choice key={m.id} selected={targetModel === m.id} onClick={() => setTargetModel(m.id)} title={m.name}>
              {m.desc}
            </Choice>
          ))}
        </div>
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
