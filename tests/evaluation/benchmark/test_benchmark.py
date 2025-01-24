from datetime import datetime
from typing import Optional, cast
from unittest.mock import Mock, patch
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel
from pytest import fixture
from requests import HTTPError, Response

from intelligence_layer.connectors.studio.studio import (
    BenchmarkLineage,
    GetBenchmarkResponse,
    PostBenchmarkExecution,
    StudioClient,
    StudioExample,
)
from intelligence_layer.evaluation.benchmark.studio_benchmark import (
    StudioBenchmarkRepository,
    create_aggregation_logic_identifier,
    create_evaluation_logic_identifier,
    type_to_schema,
)
from tests.evaluation.conftest import (
    FAIL_IN_TASK_INPUT,
    DummyAggregationLogic,
    DummyEvaluationLogic,
    DummyTask,
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
        project_id=str(uuid4()),
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
def task() -> DummyTask:
    return DummyTask()


@fixture
def evaluation_logic() -> DummyEvaluationLogic:
    return DummyEvaluationLogic()


@fixture
def aggregation_logic() -> DummyAggregationLogic:
    return DummyAggregationLogic()


def test_type_to_schema() -> None:
    class ExampleModel(BaseModel):
        name: str
        age: int

    class NestedModel(BaseModel):
        example: ExampleModel
        tags: list[str]

    assert type_to_schema(int) == {"type": "integer"}
    assert type_to_schema(str) == {"type": "string"}
    assert type_to_schema(bool) == {"type": "boolean"}
    assert type_to_schema(float) == {"type": "number"}
    assert type_to_schema(None) == {"type": "null"}  # type: ignore
    assert type_to_schema(Optional[int]) == {  # type: ignore
        "anyOf": [{"type": "integer"}, {"type": "null"}]
    }
    assert type_to_schema(list[int]) == {"type": "array", "items": {"type": "integer"}}

    assert type_to_schema(dict[str, int]) == {
        "type": "object",
        "additionalProperties": {"type": "integer"},
    }
    schema = type_to_schema(ExampleModel)
    assert schema["title"] == "ExampleModel"
    assert "properties" in schema
    assert schema["properties"]["name"]["type"] == "string"
    assert schema["properties"]["age"]["type"] == "integer"

    schema = type_to_schema(NestedModel)
    assert schema["title"] == "NestedModel"
    assert "properties" in schema
    assert "example" in schema["properties"]
    assert "tags" in schema["properties"]
    assert schema["properties"]["tags"]["type"] == "array"
    assert schema["properties"]["tags"]["items"] == {"type": "string"}


def test_extract_types_from_eval_logic(evaluation_logic: DummyEvaluationLogic) -> None:
    created_identifier = create_evaluation_logic_identifier(evaluation_logic)
    dummy_logic = "if output == FAIL_IN_EVAL_INPUT:"
    dummy_type = "DummyEvaluation"
    assert dummy_logic in created_identifier.logic
    assert dummy_type in created_identifier.logic

    assert created_identifier.input_schema["type"] == "string"
    assert created_identifier.expected_output_schema["type"] == "string"

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
    mock_submit_benchmark = cast(Mock, mock_studio_client.submit_benchmark)
    mock_submit_benchmark.return_value = str(uuid4())

    benchmark = studio_benchmark_repository.create_benchmark(
        dataset_id, evaluation_logic, aggregation_logic, "benchmark_name"
    )
    uuid = UUID(benchmark.id)
    assert uuid
    assert benchmark.dataset_id == dataset_id
    mock_submit_benchmark.assert_called_once()


def test_create_benchmark_with_non_existing_dataset(
    studio_benchmark_repository: StudioBenchmarkRepository,
    mock_studio_client: StudioClient,
    evaluation_logic: DummyEvaluationLogic,
    aggregation_logic: DummyAggregationLogic,
) -> None:
    dataset_id = "fake_dataset_id"
    response = Response()
    response.status_code = 400

    cast(Mock, mock_studio_client.submit_benchmark).side_effect = HTTPError(
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
    cast(Mock, mock_studio_client.get_benchmark).return_value = get_benchmark_response

    benchmark = studio_benchmark_repository.get_benchmark(
        benchmark_id, evaluation_logic, aggregation_logic
    )

    assert benchmark
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
    cast(Mock, mock_studio_client.get_benchmark).return_value = None

    assert (
        studio_benchmark_repository.get_benchmark(
            "non_existing_id", evaluation_logic, aggregation_logic
        )
        is None
    )


def test_create_benchmark_with_name_already_exists(
    studio_benchmark_repository: StudioBenchmarkRepository,
    mock_studio_client: StudioClient,
    evaluation_logic: DummyEvaluationLogic,
    aggregation_logic: DummyAggregationLogic,
) -> None:
    dataset_id = str(uuid4())
    response = Response()
    response.status_code = 409
    name = "benchmark_name"

    cast(Mock, mock_studio_client.submit_benchmark).side_effect = HTTPError(
        "409 Client Error: Database key constraint violated", response=response
    )

    with pytest.raises(
        ValueError,
        match=f"""Benchmark with name "{name}" already exists. Names of Benchmarks in the same Project must be unique.""",
    ):
        studio_benchmark_repository.create_benchmark(
            dataset_id, evaluation_logic, aggregation_logic, name
        )


@patch(
    "intelligence_layer.evaluation.benchmark.studio_benchmark.extract_token_count_from_trace"
)
def test_execute_benchmark(
    mock_extract_tokens: Mock,
    studio_benchmark_repository: StudioBenchmarkRepository,
    mock_studio_client: StudioClient,
    evaluation_logic: DummyEvaluationLogic,
    get_benchmark_response: GetBenchmarkResponse,
    aggregation_logic: DummyAggregationLogic,
    task: DummyTask,
) -> None:
    mock_submit_trace = cast(Mock, mock_studio_client.submit_trace)
    mock_submit_trace.return_value = str(uuid4())
    mock_submit_execution = cast(Mock, mock_studio_client.submit_benchmark_execution)
    mock_submit_lineage = cast(Mock, mock_studio_client.submit_benchmark_lineages)

    expected_generated_tokens = 100
    mock_extract_tokens.return_value = expected_generated_tokens

    cast(Mock, mock_studio_client.get_benchmark).return_value = get_benchmark_response
    examples = [
        StudioExample(input="input0", expected_output="expected_output0"),
        StudioExample(input="input1", expected_output="expected_output1"),
        StudioExample(input="input2", expected_output="expected_output2"),
        StudioExample(input="input3", expected_output="expected_output3"),
    ]
    cast(Mock, mock_studio_client.get_dataset_examples).return_value = examples
    benchmark = studio_benchmark_repository.get_benchmark(
        "benchmark_id", evaluation_logic, aggregation_logic
    )
    assert benchmark

    # when
    benchmark.execute(
        task,
        name="name",
        description="description",
        metadata={"key": "value"},
        labels={"label"},
    )

    # then
    mock_submit_execution.assert_called_once()
    uploaded_execution = cast(
        PostBenchmarkExecution, mock_submit_execution.call_args[1]["data"]
    )
    assert uploaded_execution.run_success_avg_latency > 0
    assert uploaded_execution.run_success_avg_token_count == expected_generated_tokens

    assert mock_submit_trace.call_count == 4

    mock_submit_lineage.assert_called_once()
    uploaded_lineages = mock_submit_lineage.call_args[1]["benchmark_lineages"]
    for lineage in uploaded_lineages:
        lineage = cast(BenchmarkLineage, lineage)
        assert lineage.run_latency > 0
        # this assumes that each lineage consists of traces that only have a single span
        assert lineage.run_tokens == expected_generated_tokens


def test_execute_benchmark_on_empty_examples_uploads_example_and_calculates_correctly(
    studio_benchmark_repository: StudioBenchmarkRepository,
    mock_studio_client: StudioClient,
    evaluation_logic: DummyEvaluationLogic,
    get_benchmark_response: GetBenchmarkResponse,
    aggregation_logic: DummyAggregationLogic,
    task: DummyTask,
) -> None:
    mock_submit_trace = cast(Mock, mock_studio_client.submit_trace)
    mock_submit_execution = cast(Mock, mock_studio_client.submit_benchmark_execution)

    cast(Mock, mock_studio_client.get_benchmark).return_value = get_benchmark_response
    cast(Mock, mock_studio_client.get_dataset_examples).return_value = []
    benchmark = studio_benchmark_repository.get_benchmark(
        "benchmark_id", evaluation_logic, aggregation_logic
    )
    assert benchmark

    # when
    benchmark.execute(
        task,
        name="name",
        description="description",
        metadata={"key": "value"},
        labels={"label"},
    )

    # then
    mock_submit_execution.assert_called_once()
    uploaded_execution = cast(
        PostBenchmarkExecution, mock_submit_execution.call_args[1]["data"]
    )
    assert uploaded_execution.run_success_avg_latency == 0
    assert uploaded_execution.run_success_avg_token_count == 0

    assert mock_submit_trace.call_count == 0


@patch(
    "intelligence_layer.evaluation.benchmark.studio_benchmark.extract_token_count_from_trace"
)
def test_execute_benchmark_failing_examples_calculates_correctly(
    mock_extract_tokens: Mock,
    studio_benchmark_repository: StudioBenchmarkRepository,
    mock_studio_client: StudioClient,
    evaluation_logic: DummyEvaluationLogic,
    get_benchmark_response: GetBenchmarkResponse,
    aggregation_logic: DummyAggregationLogic,
    task: DummyTask,
) -> None:
    mock_submit_trace = cast(Mock, mock_studio_client.submit_trace)
    mock_submit_execution = cast(Mock, mock_studio_client.submit_benchmark_execution)

    cast(Mock, mock_studio_client.get_benchmark).return_value = get_benchmark_response
    examples = [
        StudioExample(input=FAIL_IN_TASK_INPUT, expected_output="expected_output_0"),
    ]
    cast(Mock, mock_studio_client.get_dataset_examples).return_value = examples
    benchmark = studio_benchmark_repository.get_benchmark(
        "benchmark_id", evaluation_logic, aggregation_logic
    )

    expected_generated_tokens = 0
    mock_extract_tokens.return_value = expected_generated_tokens + 1
    assert benchmark

    # when
    benchmark.execute(
        task,
        name="name",
        description="description",
        metadata={"key": "value"},
        labels={"label"},
    )

    # then
    mock_submit_execution.assert_called_once()
    uploaded_execution = cast(
        PostBenchmarkExecution, mock_submit_execution.call_args[1]["data"]
    )
    assert uploaded_execution.run_success_avg_latency == 0
    assert uploaded_execution.run_success_avg_token_count == expected_generated_tokens
    assert uploaded_execution.run_successful_count == 0

    assert mock_submit_trace.call_count == 0
