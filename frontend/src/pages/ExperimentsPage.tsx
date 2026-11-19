import { useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { FixtureNotice } from "../components/FixtureNotice";
import { SectionHeading } from "../components/Card";
import { EmptyState } from "../components/EmptyState";
import { useExperiments, useRuns } from "../data/useData";
import { useConnection } from "../context/ConnectionContext";
import { formatTimestamp } from "../lib/derived";
import { DataStateView } from "../components/DataStateView";

const PRESET_DATASETS = [
  {
    name: "qa-geography-baseline-v1",
    title: "Geography & Fact Retrieval",
    caseCount: 5,
    description: "Evaluates exact entity extraction, capital cities, and scientific fact recall against ground truth references.",
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
    title: "RAG Evidence & Groundedness Benchmark",
    caseCount: 3,
    description: "Evaluates context faithfulness, context omission attribution, and claim grounding against retrieved passages.",
    testCases: [
      { id: "rag-1", input: "What is the capital of France?", reference_answer: "Paris", metadata: {} },
      { id: "rag-2", input: "Which fruit is high in potassium?", reference_answer: "Banana", metadata: {} },
      { id: "rag-3", input: "Who wrote Romeo and Juliet?", reference_answer: "William Shakespeare", metadata: {} },
    ],
  },
];

const AVAILABLE_MODELS = [
  {
    id: "mock-adapter-v1",
    name: "Deterministic Mock Adapter",
    category: "Local / Offline Baseline",
    desc: "Deterministic test adapter that runs locally with zero external network dependencies or API keys.",
  },
  {
    id: "rag-pipeline-v1",
    name: "RAG Pipeline (Cosine Retriever)",
    category: "Retrieval-Augmented Architecture",
    desc: "Retrieval-augmented pipeline querying a local corpus before generating responses with retrieved context.",
  },
  {
    id: "echo-adapter",
    name: "Echo Baseline Adapter",
    category: "Prompt Reproduction",
    desc: "Echoes the prompt directly for verifying baseline dataflow and normalization.",
  },
  {
    id: "openai-gpt-4o-mini",
    name: "OpenAI GPT-4o Mini",
    category: "Real Cloud Provider",
    desc: "Evaluates OpenAI's fast reasoning model. Requires external inference opt-in and valid API key.",
  },
];

const AVAILABLE_EVALUATORS = [
  {
    id: "exact_match",
    name: "Exact String Match",
    criterion: "Correctness",
    desc: "Evaluates whether the model output matches the reference answer verbatim after whitespace and casing normalization.",
  },
  {
    id: "semantic_similarity",
    name: "Semantic Similarity",
    criterion: "Semantic Equivalence",
    desc: "Measures cosine similarity over sentence embeddings to grant credit for paraphrasing and synonyms.",
  },
  {
    id: "llm_faithfulness_judge",
    name: "Faithfulness Judge",
    criterion: "Context Faithfulness",
    desc: "Applies an LLM judge to verify that statements in the answer are directly backed by retrieved context passages.",
  },
];

export function ExperimentsPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [searchInput, setSearchInput] = useState("");
  const [showCreator, setShowCreator] = useState(searchParams.get("new") === "true");

  // Creator state
  const [expName, setExpName] = useState("");
  const [expId, setExpId] = useState("");
  const [expDesc, setExpDesc] = useState("");
  const [selectedDatasetIndex, setSelectedDatasetIndex] = useState(0);
  const [targetModel, setTargetModel] = useState("mock-adapter-v1");
  const [selectedEvaluators, setSelectedEvaluators] = useState<string[]>(["exact_match"]);
  const [showAdvancedModel, setShowAdvancedModel] = useState(false);
  const [temperature, setTemperature] = useState<number>(0.0);
  const [maxTokens, setMaxTokens] = useState<number>(100);

  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const [createdExpId, setCreatedExpId] = useState<string | null>(null);

  const { dataMode, client } = useConnection();
  const { data: experiments, loading: expLoading, error: expError, refetch } = useExperiments(
    searchInput,
  );
  const { data: runs, loading: runsLoading } = useRuns();

  const loading = expLoading || runsLoading;
  const expList = experiments || [];
  const runList = runs || [];

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!expName.trim() || !expId.trim()) {
      setCreateError("Experiment Name and ID are required.");
      return;
    }
    if (selectedEvaluators.length === 0) {
      setCreateError("Select at least one evaluation method.");
      return;
    }

    setCreating(true);
    setCreateError(null);
    setCreatedExpId(null);

    const chosenDataset = PRESET_DATASETS[selectedDatasetIndex];
    const finalExpId = expId.trim();

    try {
      if (dataMode === "live") {
        await client.createExperiment({
          experiment_id: finalExpId,
          name: expName.trim(),
          description: expDesc.trim() || undefined,
          dataset_name: chosenDataset.name,
          test_cases: chosenDataset.testCases,
          model_config: {
            model_id: targetModel,
            temperature: showAdvancedModel ? temperature : 0.0,
            max_tokens: showAdvancedModel ? maxTokens : 100,
          },
          evaluator_names: selectedEvaluators,
          metadata: {},
        });
      }
      setCreatedExpId(finalExpId);
      setShowCreator(false);
      setExpName("");
      setExpId("");
      setExpDesc("");
      void refetch();
    } catch (err: unknown) {
      setCreateError(err instanceof Error ? err.message : String(err));
    } finally {
      setCreating(false);
    }
  };

  const toggleEvaluator = (evalId: string) => {
    setSelectedEvaluators((prev) =>
      prev.includes(evalId) ? prev.filter((id) => id !== evalId) : [...prev, evalId],
    );
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between border-b border-line pb-4">
        <div>
          <SectionHeading eyebrow="Benchmark Suite" title="Experiments" />
          <p className="mt-1 text-xs text-ink-soft max-w-xl">
            Controlled evaluation definitions specifying content-addressed datasets, target models, and assessment criteria.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <input
            type="text"
            placeholder="Search experiments..."
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            className="border border-line bg-paper px-3 py-1.5 text-xs text-ink placeholder:text-ink-soft focus:border-ink focus:outline-none"
          />
          <button
            type="button"
            onClick={() => {
              setShowCreator(!showCreator);
              setCreateError(null);
              setCreatedExpId(null);
            }}
            className="bg-ink px-4 py-1.5 font-mono text-xs font-semibold text-paper hover:bg-ink-soft transition-colors"
          >
            {showCreator ? "Cancel" : "+ Define Experiment"}
          </button>
        </div>
      </div>

      <FixtureNotice />

      {/* Post-Creation Success Banner with Clear Next Action */}
      {createdExpId && (
        <div className="border border-emerald-500/50 bg-emerald-950/40 p-4 text-xs text-emerald-200 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div>
            <span className="font-semibold block text-emerald-300">
              ✓ Experiment &ldquo;{createdExpId}&rdquo; created successfully!
            </span>
            <span className="text-[11px] text-emerald-400/80">
              Your evaluation benchmark is registered. The next action is to run an execution.
            </span>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => navigate(`/experiments/${createdExpId}`)}
              className="bg-emerald-400 text-black px-4 py-1.5 font-mono text-xs font-semibold hover:bg-emerald-300 transition-colors"
            >
              Run Experiment Now &rarr;
            </button>
            <button
              type="button"
              onClick={() => setCreatedExpId(null)}
              className="font-mono text-emerald-400 underline hover:text-emerald-300 px-2"
            >
              Dismiss
            </button>
          </div>
        </div>
      )}

      {/* GUIDED 3-STAGE EXPERIMENT CREATION FORM */}
      {showCreator && (
        <form onSubmit={handleCreate} className="border border-line bg-paper p-6 space-y-6">
          <div className="border-b border-line pb-3">
            <h3 className="font-display text-lg font-medium text-ink">Guided Experiment Setup</h3>
            <p className="text-xs text-ink-soft mt-0.5">
              Follow three simple stages to configure a reproducible benchmark. Advanced parameters remain collapsed by default.
            </p>
          </div>

          {createError && (
            <div className="border border-red-500/40 bg-red-950/30 p-3 text-xs text-red-200">
              {createError}
            </div>
          )}

          {/* Preliminary: Name & Identity */}
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <label htmlFor="exp-name" className="block font-mono text-xs text-ink-soft uppercase mb-1">
                Experiment Name *
              </label>
              <input
                id="exp-name"
                type="text"
                required
                placeholder="e.g. Geography Baseline Evaluation"
                value={expName}
                onChange={(e) => {
                  setExpName(e.target.value);
                  if (!expId) {
                    setExpId(
                      e.target.value
                        .toLowerCase()
                        .replace(/[^a-z0-9]+/g, "-")
                        .replace(/^-|-$/g, "")
                    );
                  }
                }}
                className="w-full border border-line bg-paper px-3 py-1.5 text-xs text-ink focus:border-ink focus:outline-none"
              />
            </div>

            <div>
              <label htmlFor="exp-id" className="block font-mono text-xs text-ink-soft uppercase mb-1">
                Experiment ID * (URL Identifier)
              </label>
              <input
                id="exp-id"
                type="text"
                required
                placeholder="e.g. exp-geography-baseline-v1"
                value={expId}
                onChange={(e) => setExpId(e.target.value)}
                className="w-full border border-line bg-paper px-3 py-1.5 font-mono text-xs text-ink focus:border-ink focus:outline-none"
              />
            </div>

            <div className="sm:col-span-2">
              <label htmlFor="exp-desc" className="block font-mono text-xs text-ink-soft uppercase mb-1">
                Research Objective (Optional)
              </label>
              <input
                id="exp-desc"
                type="text"
                placeholder="State the scientific hypothesis or evaluation purpose..."
                value={expDesc}
                onChange={(e) => setExpDesc(e.target.value)}
                className="w-full border border-line bg-paper px-3 py-1.5 text-xs text-ink focus:border-ink focus:outline-none"
              />
            </div>
          </div>

          {/* STAGE 1: CHOOSE DATASET */}
          <div className="border-t border-line pt-5 space-y-3">
            <div className="flex items-center gap-2">
              <span className="w-5 h-5 rounded-full bg-paper-dim border border-line flex items-center justify-center font-mono text-[11px] font-bold text-ink">
                1
              </span>
              <span className="font-display text-sm font-semibold text-ink uppercase tracking-wide">
                Stage 1: Choose Dataset
              </span>
            </div>
            <p className="text-xs text-ink-soft">
              Select a content-addressed test dataset containing reference answers or retrieved passages.
            </p>

            <div className="grid gap-3 sm:grid-cols-2 pt-1">
              {PRESET_DATASETS.map((ds, idx) => {
                const isSelected = selectedDatasetIndex === idx;
                return (
                  <button
                    key={ds.name}
                    type="button"
                    onClick={() => setSelectedDatasetIndex(idx)}
                    className={`text-left p-4 border transition-colors ${
                      isSelected
                        ? "border-ink bg-paper-dim shadow-xs"
                        : "border-line bg-paper hover:bg-paper-dim"
                    }`}
                  >
                    <div className="flex items-center justify-between text-xs mb-1">
                      <span className="font-display font-semibold text-ink text-sm">{ds.title}</span>
                      <span className="font-mono text-[11px] text-ink-soft bg-paper border border-line px-2 py-0.5">
                        {ds.caseCount} test cases
                      </span>
                    </div>
                    <span className="font-mono text-[11px] text-ink-soft block mb-1.5">{ds.name}</span>
                    <p className="text-xs text-ink-soft leading-relaxed">{ds.description}</p>
                  </button>
                );
              })}
            </div>
          </div>

          {/* STAGE 2: CHOOSE MODEL */}
          <div className="border-t border-line pt-5 space-y-3">
            <div className="flex items-center gap-2">
              <span className="w-5 h-5 rounded-full bg-paper-dim border border-line flex items-center justify-center font-mono text-[11px] font-bold text-ink">
                2
              </span>
              <span className="font-display text-sm font-semibold text-ink uppercase tracking-wide">
                Stage 2: Choose Target Model
              </span>
            </div>
            <p className="text-xs text-ink-soft">
              Select the language model architecture or application adapter to evaluate.
            </p>

            <div className="grid gap-3 sm:grid-cols-2 pt-1">
              {AVAILABLE_MODELS.map((m) => {
                const isSelected = targetModel === m.id;
                return (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => setTargetModel(m.id)}
                    className={`text-left p-3.5 border transition-colors ${
                      isSelected
                        ? "border-ink bg-paper-dim shadow-xs"
                        : "border-line bg-paper hover:bg-paper-dim"
                    }`}
                  >
                    <div className="flex items-center justify-between text-xs mb-0.5">
                      <span className="font-display font-semibold text-ink">{m.name}</span>
                      <span className="font-mono text-[10px] text-ink-soft">{m.category}</span>
                    </div>
                    <span className="font-mono text-[11px] text-ink-soft block mb-1">{m.id}</span>
                    <p className="text-[11px] text-ink-soft leading-relaxed">{m.desc}</p>
                  </button>
                );
              })}
            </div>

            {/* Collapsed Advanced Parameters (Progressive Disclosure) */}
            <div className="pt-2">
              <button
                type="button"
                onClick={() => setShowAdvancedModel(!showAdvancedModel)}
                className="font-mono text-xs text-ink-soft hover:text-ink flex items-center gap-1.5"
              >
                <span>{showAdvancedModel ? "▾ Hide Advanced Parameters" : "▸ Show Advanced Parameters (Temperature & Tokens)"}</span>
              </button>

              {showAdvancedModel && (
                <div className="mt-3 grid grid-cols-2 gap-4 border border-line p-4 bg-paper-dim">
                  <div>
                    <label htmlFor="temp-input" className="block font-mono text-[11px] text-ink-soft uppercase mb-1">
                      Temperature: {temperature}
                    </label>
                    <input
                      id="temp-input"
                      type="range"
                      min={0.0}
                      max={1.0}
                      step={0.1}
                      value={temperature}
                      onChange={(e) => setTemperature(parseFloat(e.target.value))}
                      className="w-full accent-ink"
                    />
                    <span className="text-[10px] text-ink-soft block mt-0.5">
                      0.0 = deterministic; 1.0 = creative/variable.
                    </span>
                  </div>
                  <div>
                    <label htmlFor="tokens-input" className="block font-mono text-[11px] text-ink-soft uppercase mb-1">
                      Max Tokens: {maxTokens}
                    </label>
                    <input
                      id="tokens-input"
                      type="number"
                      min={10}
                      max={1000}
                      value={maxTokens}
                      onChange={(e) => setMaxTokens(parseInt(e.target.value) || 100)}
                      className="w-full border border-line bg-paper px-2.5 py-1 font-mono text-xs text-ink"
                    />
                    <span className="text-[10px] text-ink-soft block mt-0.5">
                      Upper limit on output generation length.
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* STAGE 3: CHOOSE EVALUATION METHOD */}
          <div className="border-t border-line pt-5 space-y-3">
            <div className="flex items-center gap-2">
              <span className="w-5 h-5 rounded-full bg-paper-dim border border-line flex items-center justify-center font-mono text-[11px] font-bold text-ink">
                3
              </span>
              <span className="font-display text-sm font-semibold text-ink uppercase tracking-wide">
                Stage 3: Choose Evaluation Methods
              </span>
            </div>
            <p className="text-xs text-ink-soft">
              Select one or more assessment techniques. Answers are evaluated independently against each chosen method.
            </p>

            <div className="grid gap-3 sm:grid-cols-3 pt-1">
              {AVAILABLE_EVALUATORS.map((ev) => {
                const isSelected = selectedEvaluators.includes(ev.id);
                return (
                  <label
                    key={ev.id}
                    className={`flex items-start gap-3 border p-3.5 text-xs cursor-pointer transition-colors ${
                      isSelected ? "border-ink bg-paper-dim" : "border-line bg-paper hover:bg-paper-dim"
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={isSelected}
                      onChange={() => toggleEvaluator(ev.id)}
                      className="mt-0.5 accent-ink"
                    />
                    <div>
                      <div className="font-display font-semibold text-ink">{ev.name}</div>
                      <div className="font-mono text-[10px] text-ink-soft mb-1">{ev.criterion}</div>
                      <p className="text-ink-soft text-[11px] leading-relaxed">{ev.desc}</p>
                    </div>
                  </label>
                );
              })}
            </div>
          </div>

          {/* Form Actions */}
          <div className="flex items-center justify-end gap-3 border-t border-line pt-4">
            <button
              type="button"
              onClick={() => setShowCreator(false)}
              className="border border-line px-4 py-2 font-mono text-xs text-ink hover:bg-paper-dim transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={creating}
              className="bg-ink px-6 py-2 font-mono text-xs font-semibold text-paper hover:bg-ink-soft transition-colors disabled:opacity-50"
            >
              {creating ? "Creating Experiment..." : "Save & Create Experiment"}
            </button>
          </div>
        </form>
      )}

      {/* Experiments Registry Table */}
      <DataStateView loading={loading} error={expError} onRetry={refetch}>
        <div className="border border-line bg-paper overflow-hidden">
          <div className="border-b border-line bg-paper-dim px-4 py-2.5 flex items-center justify-between text-xs font-mono text-ink-soft">
            <span>Defined Experiments ({expList.length})</span>
            <span>Sorted chronologically</span>
          </div>

          {expList.length === 0 ? (
            <EmptyState
              title={searchInput ? "No experiments match your search query." : "No experiments defined yet."}
              detail={searchInput ? "Try clearing your search term or define a new experiment." : "Create your first controlled experiment using the button above."}
            />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-sans">
                <thead>
                  <tr className="border-b border-line bg-paper-dim uppercase font-mono text-[11px] text-ink-soft">
                    <th className="p-3">Experiment</th>
                    <th className="p-3">Dataset</th>
                    <th className="p-3">Model</th>
                    <th className="p-3">Evaluators</th>
                    <th className="p-3">Created</th>
                    <th className="p-3 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line">
                  {expList.map((experiment) => {
                    const expRuns = runList.filter((r) => r.experiment_id === experiment.experiment_id);
                    const latest = expRuns[0];

                    return (
                      <tr key={experiment.experiment_id} className="hover:bg-paper-dim transition-colors">
                        <td className="p-3">
                          <Link
                            to={`/experiments/${experiment.experiment_id}`}
                            className="font-medium text-ink hover:underline text-sm"
                          >
                            {experiment.name}
                          </Link>
                          <span className="block font-mono text-[10px] text-ink-soft">
                            {experiment.experiment_id}
                          </span>
                        </td>
                        <td className="p-3">
                          <span className="font-mono text-xs text-ink font-medium">
                            {experiment.dataset_version?.dataset_name || "N/A"}
                          </span>
                          <span className="block text-[11px] text-ink-soft">
                            {experiment.dataset_version?.test_case_count ??
                              experiment.dataset_version?.test_cases?.length ??
                              experiment.test_case_count ??
                              0}{" "}
                            test cases
                          </span>
                        </td>
                        <td className="p-3 font-mono text-xs text-ink">
                          {experiment.model_config?.model_id || "N/A"}
                        </td>
                        <td className="p-3 font-mono text-xs text-ink-soft">
                          {experiment.evaluator_configs?.map((e) => e.evaluator_name).join(", ") || "None"}
                        </td>
                        <td className="p-3 text-ink-soft text-[11px]">
                          {formatTimestamp(experiment.created_at)}
                        </td>
                        <td className="p-3 text-right font-mono text-xs space-x-2">
                          <Link
                            to={`/experiments/${experiment.experiment_id}`}
                            className="bg-paper-dim border border-line px-2.5 py-1 text-ink hover:bg-paper inline-block"
                          >
                            Run / View
                          </Link>
                          {latest && (
                            <Link
                              to={`/results/${latest.run_id}`}
                              className="text-ink underline hover:text-ink-soft inline-block"
                            >
                              Results &rarr;
                            </Link>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </DataStateView>
    </div>
  );
}
