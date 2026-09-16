"""
Minimal demonstration of the Core Evaluation Engine.

Runs a tiny, hand-written dataset through the deterministic MockAdapter and
the baseline ExactMatchEvaluator, then prints a structured, human-readable
summary followed by the full machine-readable JSON result.

Requires no network access, no API credentials, and no external services.
The MockAdapter's outputs are fake and hard-coded for demonstration only;
they do not represent real model behavior.

Usage:
    python scripts/run_core_engine_example.py
"""

from __future__ import annotations

import sys

from llm_reliability.evaluation import (
    Dataset,
    EvaluationRunner,
    ExactMatchEvaluator,
    MockAdapter,
    ModelConfig,
    RunResult,
    TestCase,
)


def build_example_dataset() -> Dataset:
    return Dataset(
        name="core-engine-demo",
        test_cases=[
            TestCase(
                id="capital-of-france",
                input="What is the capital of France?",
                reference_answer="Paris",
            ),
            TestCase(
                id="capital-of-japan",
                input="What is the capital of Japan?",
                reference_answer="Tokyo",
            ),
            TestCase(
                id="open-ended-opinion",
                input="What is the most interesting unsolved problem in mathematics?",
                # No reference_answer: this test case demonstrates reference-free
                # evaluation, since exact-match is not applicable to open-ended answers.
            ),
        ],
    )


def build_example_adapter() -> MockAdapter:
    # A real adapter would call a provider API here. The mock adapter instead
    # returns pre-scripted, deterministic text so the example is reproducible
    # without any external dependency. "Osaka" is deliberately wrong to
    # demonstrate how a failed evaluation is reported.
    return MockAdapter(
        responses={
            "capital-of-france": "Paris",
            "capital-of-japan": "Osaka",
            "open-ended-opinion": "The Riemann Hypothesis remains unsolved and highly influential.",
        }
    )


def print_summary(run_result: RunResult) -> None:
    print(f"Dataset: {run_result.dataset_name}")
    print(f"Model: {run_result.model_id}")
    print(f"Evaluators: {', '.join(run_result.evaluator_names)}")
    print(
        f"Test cases: {run_result.num_test_cases} "
        f"(succeeded: {run_result.num_succeeded}, failed: {run_result.num_failed})"
    )
    print()

    for result in run_result.test_case_results:
        print(f"[{result.test_case_id}]")
        print(f"  input: {result.input_text}")
        print(f"  reference_answer: {result.reference_answer!r}")
        if result.model_response is None:
            print(f"  EXECUTION ERROR: {result.execution_error}")
            continue
        print(f"  model_output: {result.model_response.output_text!r}")
        for evaluation in result.evaluation_results:
            print(
                f"  evaluator={evaluation.evaluator_name} "
                f"label={evaluation.label} passed={evaluation.passed} score={evaluation.score}"
            )
        print()


def main() -> int:
    dataset = build_example_dataset()
    adapter = build_example_adapter()
    runner = EvaluationRunner(adapter=adapter, evaluators=[ExactMatchEvaluator()])

    run_result = runner.run(dataset, ModelConfig(model_id=adapter.model_id))

    print_summary(run_result)
    print("--- Full structured result (JSON) ---")
    print(run_result.to_json())
    return 0


if __name__ == "__main__":
    sys.exit(main())
