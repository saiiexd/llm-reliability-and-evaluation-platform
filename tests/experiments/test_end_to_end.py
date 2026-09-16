"""End-to-end test of the Experiment System.

Covers the full pipeline required for this milestone:
Experiment Definition -> Execution -> Persistence -> Reload.
"""

from llm_reliability.evaluation import (
    Dataset,
    ExactMatchEvaluator,
    MockAdapter,
    ModelConfig,
    TestCase,
)
from llm_reliability.experiments.local_repository import LocalFileExperimentRepository
from llm_reliability.experiments.manager import ExperimentManager
from llm_reliability.experiments.models import RunStatus


def test_experiment_definition_through_execution_persistence_and_reload(tmp_path):
    dataset = Dataset(
        name="capitals",
        test_cases=[
            TestCase(id="capital-france", input="Capital of France?", reference_answer="Paris"),
            TestCase(id="capital-japan", input="Capital of Japan?", reference_answer="Tokyo"),
        ],
    )
    model_config = ModelConfig(model_id="mock-adapter-v1")
    evaluator = ExactMatchEvaluator()

    manager = ExperimentManager(LocalFileExperimentRepository(tmp_path))

    definition = manager.create_experiment(
        experiment_id="exp-capitals-baseline",
        name="Capitals baseline",
        dataset=dataset,
        model_config=model_config,
        evaluators=[evaluator],
        description="Baseline exact-match check against a tiny capitals dataset.",
    )

    # Both test cases execute successfully; only the evaluation of one of them
    # (Osaka vs. the reference "Tokyo") reports a mismatch. Run status reflects
    # execution outcome, not evaluation correctness, so this run still succeeds.
    adapter = MockAdapter(responses={"capital-france": "Paris", "capital-japan": "Osaka"})
    run = manager.execute_experiment(definition, adapter, [evaluator])

    assert run.status == RunStatus.SUCCEEDED
    assert run.experiment_id == definition.experiment_id

    # Reload both records as if in a fresh process, with no in-memory state left.
    fresh_manager = ExperimentManager(LocalFileExperimentRepository(tmp_path))
    reloaded_definition = fresh_manager.load_experiment("exp-capitals-baseline")
    reloaded_run = fresh_manager.load_run(run.run_id)

    assert reloaded_definition.config_fingerprint == definition.config_fingerprint
    assert reloaded_definition.dataset_version.version_id == definition.dataset_version.version_id

    assert reloaded_run.status == RunStatus.SUCCEEDED
    assert reloaded_run.config_fingerprint == run.config_fingerprint

    by_id = {r.test_case_id: r for r in reloaded_run.result.test_case_results}
    assert by_id["capital-france"].evaluation_results[0].passed is True
    assert by_id["capital-japan"].evaluation_results[0].passed is False

    # A second execution of the same experiment gets a new run identity but
    # the same experiment identity and the same configuration fingerprint.
    second_run = fresh_manager.execute_experiment(reloaded_definition, adapter, [evaluator])
    assert second_run.run_id != run.run_id
    assert second_run.experiment_id == run.experiment_id
    assert second_run.config_fingerprint == run.config_fingerprint
