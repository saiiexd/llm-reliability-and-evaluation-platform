"""
Minimal demonstration of the Experiment System.

Walks through the full reproducibility workflow on top of the Core
Evaluation Engine: construct a dataset and configuration, define an
experiment, execute it through EvaluationRunner via ExperimentManager using
the deterministic MockAdapter, persist both the experiment definition and
the run locally, then reload each from disk (as if in a fresh process) and
inspect the reloaded, validated data.

Requires no network access, no API credentials, and no external services.
Persisted files are written to a temporary directory that is removed when
the script exits; this script is a demonstration of the API, not a
long-lived experiment store.

Usage:
    python scripts/run_experiment_example.py
"""

from __future__ import annotations

import sys
import tempfile

from llm_reliability.evaluation import (
    Dataset,
    ExactMatchEvaluator,
    MockAdapter,
    ModelConfig,
    TestCase,
)
from llm_reliability.experiments import (
    ExperimentDefinition,
    ExperimentManager,
    ExperimentRun,
    LocalFileExperimentRepository,
)


def build_example_dataset() -> Dataset:
    return Dataset(
        name="capitals-baseline",
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
        ],
    )


def build_example_adapter() -> MockAdapter:
    # "Osaka" is deliberately wrong, to show a mismatched evaluation inside
    # an otherwise successfully executed run.
    return MockAdapter(
        responses={"capital-of-france": "Paris", "capital-of-japan": "Osaka"},
        model_id="mock-adapter-v1",
    )


def print_definition_summary(definition: ExperimentDefinition) -> None:
    print(f"Experiment: {definition.experiment_id} ({definition.name!r})")
    print(f"  dataset_version_id: {definition.dataset_version.version_id}")
    print(f"  model_config: {definition.model_config}")
    print(f"  evaluator_configs: {[ec.to_dict() for ec in definition.evaluator_configs]}")
    print(f"  config_fingerprint: {definition.config_fingerprint}")
    print()


def print_run_summary(run: ExperimentRun) -> None:
    print(f"Run: {run.run_id} (experiment: {run.experiment_id})")
    print(f"  status: {run.status}")
    print(f"  started_at: {run.started_at}")
    print(f"  finished_at: {run.finished_at}")
    print(f"  config_fingerprint: {run.config_fingerprint}")
    if run.result is not None:
        for result in run.result.test_case_results:
            outcome = result.evaluation_results[0] if result.evaluation_results else None
            output_text = (
                result.model_response.output_text
                if result.model_response is not None
                else f"<execution error: {result.execution_error}>"
            )
            print(
                f"  [{result.test_case_id}] output={output_text!r} "
                f"passed={outcome.passed if outcome else 'n/a'}"
            )
    print()


def main() -> int:
    dataset = build_example_dataset()
    model_config = ModelConfig(model_id="mock-adapter-v1")
    evaluator = ExactMatchEvaluator()

    with tempfile.TemporaryDirectory(prefix="llm-reliability-experiment-store-") as store_dir:
        print(f"Local experiment store: {store_dir}\n")

        manager = ExperimentManager(LocalFileExperimentRepository(store_dir))

        # 1. Define the experiment. This persists the definition immediately.
        definition = manager.create_experiment(
            experiment_id="exp-capitals-baseline-v1",
            name="Capitals baseline (exact match)",
            dataset=dataset,
            model_config=model_config,
            evaluators=[evaluator],
            description=(
                "Baseline exact-match check of a tiny capitals dataset, "
                "used only to validate the experiment execution infrastructure."
            ),
        )
        print("--- Experiment created and persisted ---")
        print_definition_summary(definition)

        # 2. Execute it through the existing EvaluationRunner (via the manager).
        adapter = build_example_adapter()
        run = manager.execute_experiment(definition, adapter, [evaluator])
        print("--- Experiment executed and run persisted ---")
        print_run_summary(run)

        # 3. Reload both records from disk, as a fresh process would.
        fresh_manager = ExperimentManager(LocalFileExperimentRepository(store_dir))
        reloaded_definition = fresh_manager.load_experiment("exp-capitals-baseline-v1")
        reloaded_run = fresh_manager.load_run(run.run_id)

        print("--- Reloaded from persistence (fresh ExperimentManager instance) ---")
        print_definition_summary(reloaded_definition)
        print_run_summary(reloaded_run)

        fingerprints_match = reloaded_definition.config_fingerprint == definition.config_fingerprint
        print(f"Reloaded configuration fingerprint matches original: {fingerprints_match}")

        # 4. Execute the same experiment a second time: same experiment
        #    identity and fingerprint, but a distinct run identity.
        second_run = fresh_manager.execute_experiment(reloaded_definition, adapter, [evaluator])
        print(
            f"\nSecond run id: {second_run.run_id} (first run id: {run.run_id}, "
            f"different: {second_run.run_id != run.run_id})"
        )
        print(
            "Both runs share experiment_id "
            f"'{second_run.experiment_id}' and config_fingerprint "
            f"'{second_run.config_fingerprint}'."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
