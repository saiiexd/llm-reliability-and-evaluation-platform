import { describe, expect, it } from "vitest";
import { defaultAdapterFor, liveJudgeModels } from "../pages/ExperimentDetailPage";
import type { ExperimentDefinition } from "../types/domain";

function exp(extra: Record<string, unknown>, judges: Record<string, unknown>[] = []): ExperimentDefinition {
  return {
    experiment_id: "e",
    model_config: { model_id: "m", extra_params: extra },
    evaluator_configs: judges.map((parameters) => ({ evaluator_name: "llm_judge", parameters })),
  } as unknown as ExperimentDefinition;
}

describe("defaultAdapterFor", () => {
  it("matches the experiment's recorded configuration", () => {
    expect(defaultAdapterFor(exp({}))).toBe("mock-adapter-v1");
    expect(defaultAdapterFor(exp({ rag: { top_k: 1, embedding_model_id: "fake-embedding-v1" } }))).toBe(
      "rag-pipeline-v1",
    );
    expect(defaultAdapterFor(exp({ rag: { top_k: 2, embedding_model_id: "fake-embedding-v1" } }))).toBe(
      "rag-pipeline-topk-2",
    );
    expect(defaultAdapterFor(exp({ rag: { top_k: 3, embedding_model_id: "nomic-embed-text" } }))).toBe(
      "openai-custom-rag",
    );
    expect(defaultAdapterFor(exp({ prompt_template_id: "closed_book_concise_v1" }))).toBe("openai-custom");
  });

  it("treats only non-fake judge models as live", () => {
    const e = exp({}, [{ judge_model_id: "fake-judge-v1" }, { judge_model_id: "qwen3:4b-instruct" }]);
    expect(liveJudgeModels(e)).toEqual(["qwen3:4b-instruct"]);
    expect(defaultAdapterFor(e)).toBe("openai-custom");
  });
});
