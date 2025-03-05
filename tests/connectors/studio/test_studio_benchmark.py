from collections.abc import Iterable, Sequence
from datetime import datetime
from http import HTTPStatus
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel
from pytest import fixture
from requests import HTTPError

from intelligence_layer.connectors.studio.studio import (
    AggregationLogicIdentifier,
    BenchmarkLineage,
    EvaluationLogicIdentifier,
    PostBenchmarkExecution,
    PostBenchmarkLineagesRequest,
    StudioClient,
    StudioDataset,
    StudioExample,
)
from intelligence_layer.core.tracer.in_memory_tracer import InMemoryTracer
from intelligence_layer.core.tracer.tracer import ExportedSpan
from intelligence_layer.evaluation.aggregation.aggregator import AggregationLogic
from intelligence_layer.evaluation.benchmark.studio_benchmark import (
    create_aggregation_logic_identifier,
    create_evaluation_logic_identifier,
)
from intelligence_layer.evaluation.dataset.domain import Example
from intelligence_layer.evaluation.evaluation.evaluator.evaluator import (
    SingleOutputEvaluationLogic,
)
from tests.connectors.studio.test_studio import TracerTestTask


class DummyInput(BaseModel):
    input_data: dict[str, int]


class DummyExpectedOutput(BaseModel):
    expected_output: dict[str, int]


class DummyOutput(BaseModel):
    output: dict[str, DummyInput]


class DummyExample(Example[DummyInput, DummyExpectedOutput]):
    pass


class DummyEvaluation(BaseModel):
    input: DummyInput
    expected: DummyExpectedOutput
    output: DummyOutput


class DummyAggregation(BaseModel):
    num_evaluations: int
    descriptors: str


class DummyAggregatedEvaluation(BaseModel):
    score: dict[str, float]


class DummyEvaluationLogic(
    SingleOutputEvaluationLogic[
        DummyInput,
        DummyOutput,
        DummyExpectedOutput,
        DummyEvaluation,
    ]
):
    def do_evaluate_single_output(
        self,
        example: Example[DummyInput, DummyExpectedOutput],
        output: DummyOutput,
    ) -> DummyEvaluation:
        return DummyEvaluation(
            input=example.input, expected=example.expected_output, output=output
        )


class DummyAggregationLogic(AggregationLogic[DummyEvaluation, DummyAggregation]):
    def aggregate(self, evaluations: Iterable[DummyEvaluation]) -> DummyAggregation:
        return DummyAggregation(
            num_evaluations=len(list(evaluations)),
            descriptors="".join(str(eval.output) for eval in evaluations),
        )


class DummyBenchmarkLineage(
    BenchmarkLineage[DummyInput, DummyExpectedOutput, DummyOutput, DummyEvaluation]
):
    pass


class DummyPostBenchmarkLineagesRequest(PostBenchmarkLineagesRequest):
    pass


@fixture
def studio_dataset(
    studio_client: StudioClient, examples: Sequence[StudioExample[str, str]]
) -> str:
    return studio_client.submit_dataset(StudioDataset(name="dataset_name"), examples)


@fixture
def post_benchmark_execution() -> PostBenchmarkExecution:
    return PostBenchmarkExecution(
        name="name",
        description="Test benchmark execution",
        labels={"performance", "testing"},
        metadata={"project": "AI Testing", "team": "QA"},
        start=datetime.now(),
        end=datetime.now(),
        run_start=datetime.now(),
        run_end=datetime.now(),
        run_successful_count=10,
        run_failed_count=2,
        run_success_avg_latency=120,
        run_success_avg_token_count=300,
        eval_start=datetime.now(),
        eval_end=datetime.now(),
        eval_successful_count=8,
        eval_failed_count=1,
        aggregation_start=datetime.now(),
        aggregation_end=datetime.now(),
        statistics=DummyAggregation(
            num_evaluations=1, descriptors="empty"
        ).model_dump_json(),
    )


@fixture
def evaluation_logic_identifier() -> EvaluationLogicIdentifier:
    return create_evaluation_logic_identifier(DummyEvaluationLogic())


@fixture
def aggregation_logic_identifier() -> AggregationLogicIdentifier:
    return create_aggregation_logic_identifier(DummyAggregationLogic())


@fixture
def test_trace() -> Sequence[ExportedSpan]:
    tracer = InMemoryTracer()
    task = TracerTestTask()
    task.run("my input", tracer)
    return tracer.export_for_viewing()


@fixture
def with_uploaded_test_trace(
    test_trace: Sequence[ExportedSpan], studio_client: StudioClient
) -> str:
    trace_id = studio_client.submit_trace(test_trace)
    return trace_id


@fixture
def with_uploaded_benchmark(
    studio_client: StudioClient,
    studio_dataset: str,
    evaluation_logic_identifier: EvaluationLogicIdentifier,
    aggregation_logic_identifier: AggregationLogicIdentifier,
) -> str:
    benchmark_id = studio_client.submit_benchmark(
        studio_dataset,
        evaluation_logic_identifier,
        aggregation_logic_identifier,
        "benchmark_name",
    )
    return benchmark_id


@fixture
def with_uploaded_benchmark_execution(
    studio_client: StudioClient,
    studio_dataset: str,
    evaluation_logic_identifier: EvaluationLogicIdentifier,
    aggregation_logic_identifier: AggregationLogicIdentifier,
    with_uploaded_benchmark: str,
    post_benchmark_execution: PostBenchmarkExecution,
) -> str:
    benchmark_execution_id = studio_client.submit_benchmark_execution(
        benchmark_id=with_uploaded_benchmark, data=post_benchmark_execution
    )
    return benchmark_execution_id


def dummy_lineage(
    trace_id: str, input: str = "input", output: str = "output"
) -> DummyBenchmarkLineage:
    created_input = DummyInput(input_data={input: 1})
    return DummyBenchmarkLineage(
        trace_id=trace_id,
        input=created_input,
        expected_output=DummyExpectedOutput(expected_output={"expected": 2}),
        example_metadata={"key3": "value3"},
        output=DummyOutput(output={output: created_input}),
        evaluation={"key5": "value5"},
        run_latency=1,
        run_tokens=3,
    )


def test_create_benchmark(
    studio_client: StudioClient,
    studio_dataset: str,
    evaluation_logic_identifier: EvaluationLogicIdentifier,
    aggregation_logic_identifier: AggregationLogicIdentifier,
) -> None:
    benchmark_id = studio_client.submit_benchmark(
        studio_dataset,
        evaluation_logic_identifier,
        aggregation_logic_identifier,
        "benchmark_name",
    )
    uuid = UUID(benchmark_id)
    assert uuid


def test_create_benchmark_with_non_existing_dataset(
    studio_client: StudioClient,
    evaluation_logic_identifier: EvaluationLogicIdentifier,
    aggregation_logic_identifier: AggregationLogicIdentifier,
) -> None:
    with pytest.raises(HTTPError, match=str(HTTPStatus.NOT_FOUND.value)):
        studio_client.submit_benchmark(
            str(uuid4()),
            evaluation_logic_identifier,
            aggregation_logic_identifier,
            "benchmark_name",
        )


def test_get_benchmark(
    studio_client: StudioClient,
    studio_dataset: str,
    with_uploaded_benchmark: str,
) -> None:
    benchmark_name = "benchmark_name"

    benchmark_id = with_uploaded_benchmark

    benchmark = studio_client.get_benchmark(benchmark_id)
    assert benchmark
    assert benchmark.dataset_id == studio_dataset
    assert "DummyEvaluation" in benchmark.evaluation_logic.logic
    assert benchmark.project_id == studio_client.project_id
    assert benchmark.name == benchmark_name


def test_get_non_existing_benchmark(studio_client: StudioClient) -> None:
    benchmark = studio_client.get_benchmark(str(uuid4()))
    assert not benchmark


def test_can_create_benchmark_execution(
    studio_client: StudioClient,
    with_uploaded_benchmark: str,
    post_benchmark_execution: PostBenchmarkExecution,
) -> None:
    benchmark_id = with_uploaded_benchmark

    example_request = post_benchmark_execution

    benchmark_execution_id = studio_client.submit_benchmark_execution(
        benchmark_id=benchmark_id, data=example_request
    )

    assert UUID(benchmark_execution_id)


def test_can_submit_lineages(
    studio_client: StudioClient,
    with_uploaded_test_trace: str,
    with_uploaded_benchmark: str,
    with_uploaded_benchmark_execution: str,
    post_benchmark_execution: PostBenchmarkExecution,
) -> None:
    trace_id = with_uploaded_test_trace
    benchmark_id = with_uploaded_benchmark
    benchmark_execution_id = with_uploaded_benchmark_execution

    lineages = [
        dummy_lineage(
            trace_id,
        ),
        dummy_lineage(trace_id, "slightly longer input", "slightly_longer_output"),
    ]

    lineage_ids = studio_client.submit_benchmark_lineages(
        benchmark_lineages=lineages,
        benchmark_id=benchmark_id,
        execution_id=benchmark_execution_id,
        max_payload_size=len(lineages[1].model_dump_json().encode("utf-8"))
        + 1,  # to enforce making to requests for the lineages
    )

    assert len(lineage_ids.root) == len(lineages)
    for lineage_id in lineage_ids.root:
        assert UUID(lineage_id)


def test_submit_lineage_skips_lineages_exceeding_request_size(
    studio_client: StudioClient,
    with_uploaded_test_trace: str,
    with_uploaded_benchmark: str,
    with_uploaded_benchmark_execution: str,
) -> None:
    trace_id = with_uploaded_test_trace
    benchmark_id = with_uploaded_benchmark
    benchmark_execution_id = with_uploaded_benchmark_execution

    lineages = [
        dummy_lineage(trace_id),
        dummy_lineage(
            trace_id,
            input="input input2 input3 input4 input5",
            output="output output output output",
        ),
    ]

    lineage_ids = studio_client.submit_benchmark_lineages(
        benchmark_lineages=lineages,
        benchmark_id=benchmark_id,
        execution_id=benchmark_execution_id,
        max_payload_size=len(lineages[0].model_dump_json().encode("utf-8"))
        + 1,  # to enforce second lineage exceeds
    )

    fetched_lineage = studio_client.get_benchmark_lineage(
        benchmark_id=benchmark_id,
        execution_id=benchmark_execution_id,
        lineage_id=lineage_ids.root[0],
    )
    assert len(lineage_ids.root) == 1
    assert fetched_lineage
    assert fetched_lineage.input == lineages[0].input.model_dump()
