from uuid import UUID, uuid4

from intelligence_layer.connectors.studio.studio import StudioClient
from intelligence_layer.evaluation.benchmark.studio_benchmark import (
    StudioBenchmarkRepository,
    create_aggregation_logic_identifier,
    create_evaluation_logic_identifier,
)
from tests.evaluation.conftest import (
    DummyAggregationLogic,
    DummyEvaluationLogic,
)


def test_extract_types_from_eval_logic() -> None:
    eval_logic = DummyEvaluationLogic()
    created_identifier = create_evaluation_logic_identifier(eval_logic)
    dummy_logic = "if output == FAIL_IN_EVAL_INPUT:"
    dummy_type = "DummyEvaluation"
    assert dummy_logic in created_identifier.logic
    assert dummy_type in created_identifier.logic

    assert created_identifier.input_schema["type"] == "string"
    assert created_identifier.expected_output_schema["type"] == "null"

    assert created_identifier.evaluation_schema["type"] == "object"
    assert (
        created_identifier.evaluation_schema["properties"]["result"]["type"] == "string"
    )


def test_extract_types_from_aggregation_logic() -> None:
    aggregation_logic = DummyAggregationLogic()
    created_identifier = create_aggregation_logic_identifier(aggregation_logic)
    dummy_logic = "return DummyAggregation(num_evaluations=len(list(evaluations)))"

    assert dummy_logic in created_identifier.logic
    assert created_identifier.evaluation_schema["type"] == "object"
    assert created_identifier.aggregation_schema["type"] == "object"


def test_create_benchmark(
    studio_benchmark_repository: StudioBenchmarkRepository, mock_studio_client: StudioClient
) -> None:
    eval_logic = DummyEvaluationLogic()
    aggregation_logic = DummyAggregationLogic()
    dataset_id = "fake_dataset_id"
    mock_studio_client.create_benchmark.return_value = str(uuid4())

    benchmark = studio_benchmark_repository.create_benchmark(
        dataset_id, eval_logic, aggregation_logic, "benchmark_name"
    )
    uuid = UUID(benchmark.id)
    assert uuid
    assert benchmark.dataset_id == dataset_id
    studio_benchmark_repository.client.create_benchmark.assert_called_once()


def test_get_benchmark(
    studio_benchmark_repository: StudioBenchmarkRepository, mock_studio_client: StudioClient
) -> None:
    eval_logic = DummyEvaluationLogic()
    aggregation_logic = DummyAggregationLogic()
    mock_studio_client.create_benchmark.return_value = str(uuid4())


    benchmark = studio_benchmark_repository.get_benchmark(
        "benchmark_id", eval_logic, aggregation_logic
    )
    assert benchmark.id == "benchmark_id"
    assert benchmark.dataset_id == "dataset_id"
    assert benchmark.eval_logic
    assert benchmark.aggregation_logic


def test_get_non_existing_benchmark(
    studio_benchmark_repository: StudioBenchmarkRepository,
) -> None:
    eval_logic = DummyEvaluationLogic()
    aggregation_logic = DummyAggregationLogic()

    studio_benchmark_repository.get_benchmark(
        "non_existing_id", eval_logic, aggregation_logic
    )
