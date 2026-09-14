"""End-to-end test of the Core Evaluation Engine.

Covers the full pipeline explicitly required for this milestone:
Dataset -> Mock Adapter -> Evaluator -> Evaluation Result.
"""

from llm_reliability.evaluation import (
    Dataset,
    EvaluationRunner,
    ExactMatchEvaluator,
    ModelConfig,
    MockAdapter,
    TestCase,
)


def test_dataset_through_mock_adapter_and_evaluator_produces_structured_results():
    dataset = Dataset(
        name="capitals",
        test_cases=[
            TestCase(id="capital-france", input="What is the capital of France?", reference_answer="Paris"),
            TestCase(id="capital-japan", input="What is the capital of Japan?", reference_answer="Tokyo"),
            TestCase(id="open-ended", input="Describe the weather today."),
        ],
    )

    adapter = MockAdapter(
        responses={
            "capital-france": "Paris",
            "capital-japan": "Osaka",
        }
    )
    runner = EvaluationRunner(adapter=adapter, evaluators=[ExactMatchEvaluator()])
    run_result = runner.run(dataset, ModelConfig(model_id="mock-adapter-v1"))

    assert run_result.dataset_name == "capitals"
    assert run_result.num_test_cases == 3
    assert run_result.num_succeeded == 3
    assert run_result.num_failed == 0

    by_id = {r.test_case_id: r for r in run_result.test_case_results}

    correct = by_id["capital-france"]
    assert correct.model_response.output_text == "Paris"
    assert correct.evaluation_results[0].passed is True
    assert correct.evaluation_results[0].score == 1.0

    incorrect = by_id["capital-japan"]
    assert incorrect.model_response.output_text == "Osaka"
    assert incorrect.evaluation_results[0].passed is False
    assert incorrect.evaluation_results[0].score == 0.0

    no_reference = by_id["open-ended"]
    assert no_reference.evaluation_results[0].label == "no_reference"
    assert no_reference.evaluation_results[0].passed is None

    serialized = run_result.to_json()
    assert "capital-france" in serialized
