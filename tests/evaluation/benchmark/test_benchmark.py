from datetime import datetime
from uuid import UUID, uuid4

import pytest
from pytest import fixture
from requests import HTTPError, Response

from intelligence_layer.connectors.studio.studio import (
    GetBenchmarkResponse,
    StudioClient,
)
from intelligence_layer.evaluation.benchmark.studio_benchmark import (
    StudioBenchmarkRepository,
    create_aggregation_logic_identifier,
    create_evaluation_logic_identifier,
)
from tests.evaluation.conftest import (
    DummyAggregationLogic,
    DummyEvaluationLogic,
)


@fixture
def datatset_id() -> str:
    return "dataset_id"


@fixture
def get_benchmark_response(datatset_id: str) -> GetBenchmarkResponse:
    eval_logic = DummyEvaluationLogic()
    evaluation_identifier = create_evaluation_logic_identifier(eval_logic)
    aggregation_logic = DummyAggregationLogic()
    aggregation_identifier = create_aggregation_logic_identifier(aggregation_logic)
    return GetBenchmarkResponse(
        id="id",
        project_id=0,
        dataset_id=datatset_id,
        name="name",
        description="description",
        benchmark_metadata=None,
        evaluation_logic=evaluation_identifier,
        aggregation_logic=aggregation_identifier,
        created_at=datetime.now(),
        updated_at=None,
        last_executed_at=None,
        created_by=None,
        updated_by=None,
    )


@fixture
def evaluation_logic() -> DummyEvaluationLogic:
    return DummyEvaluationLogic()


@fixture
def aggregation_logic() -> DummyAggregationLogic:
    return DummyAggregationLogic()


def test_extract_types_from_eval_logic(evaluation_logic: DummyEvaluationLogic) -> None:
    created_identifier = create_evaluation_logic_identifier(evaluation_logic)
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


def test_extract_types_from_aggregation_logic(
    aggregation_logic: DummyAggregationLogic,
) -> None:
    created_identifier = create_aggregation_logic_identifier(aggregation_logic)
    dummy_logic = "return DummyAggregation(num_evaluations=len(list(evaluations)))"

    assert dummy_logic in created_identifier.logic
    assert created_identifier.evaluation_schema["type"] == "object"
    assert created_identifier.aggregation_schema["type"] == "object"


def test_create_benchmark(
    studio_benchmark_repository: StudioBenchmarkRepository,
    mock_studio_client: StudioClient,
    evaluation_logic: DummyEvaluationLogic,
    aggregation_logic: DummyAggregationLogic,
) -> None:
    dataset_id = "fake_dataset_id"
    mock_studio_client.create_benchmark.return_value = str(uuid4())  # type: ignore

    benchmark = studio_benchmark_repository.create_benchmark(
        dataset_id, evaluation_logic, aggregation_logic, "benchmark_name"
    )
    uuid = UUID(benchmark.id)
    assert uuid
    assert benchmark.dataset_id == dataset_id
    studio_benchmark_repository.client.create_benchmark.assert_called_once()  # type: ignore


def test_create_benchmark_with_non_existing_dataset(
    studio_benchmark_repository: StudioBenchmarkRepository,
    mock_studio_client: StudioClient,
    evaluation_logic: DummyEvaluationLogic,
    aggregation_logic: DummyAggregationLogic,
) -> None:
    dataset_id = "fake_dataset_id"
    response = Response()
    response.status_code = 400

    mock_studio_client.create_benchmark.side_effect = HTTPError(  # type: ignore
        "400 Client Error: Bad Request for url", response=response
    )

    with pytest.raises(ValueError, match=f"Dataset with ID {dataset_id} not found"):
        studio_benchmark_repository.create_benchmark(
            dataset_id, evaluation_logic, aggregation_logic, "benchmark_name"
        )


def test_get_benchmark(
    studio_benchmark_repository: StudioBenchmarkRepository,
    mock_studio_client: StudioClient,
    get_benchmark_response: GetBenchmarkResponse,
    evaluation_logic: DummyEvaluationLogic,
    aggregation_logic: DummyAggregationLogic,
    datatset_id: str,
) -> None:
    benchmark_id = "benchmark_id"
    mock_studio_client.get_benchmark.return_value = get_benchmark_response  # type: ignore

    benchmark = studio_benchmark_repository.get_benchmark(
        benchmark_id, evaluation_logic, aggregation_logic
    )
    assert benchmark.id == benchmark_id
    assert benchmark.dataset_id == datatset_id
    assert benchmark.eval_logic
    assert benchmark.aggregation_logic


def test_get_non_existing_benchmark(
    studio_benchmark_repository: StudioBenchmarkRepository,
    mock_studio_client: StudioClient,
    evaluation_logic: DummyEvaluationLogic,
    aggregation_logic: DummyAggregationLogic,
) -> None:
    mock_studio_client.get_benchmark.return_value = None  # type: ignore

    with pytest.raises(ValueError, match="Benchmark not found"):
        studio_benchmark_repository.get_benchmark(
            "non_existing_id", evaluation_logic, aggregation_logic
        )
