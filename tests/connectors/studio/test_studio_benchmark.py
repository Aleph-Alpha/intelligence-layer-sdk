from collections.abc import Iterable, Sequence
from http import HTTPStatus
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel
from pytest import fixture
from requests import HTTPError

from intelligence_layer.connectors.studio.studio import (
    AggregationLogicIdentifier,
    EvaluationLogicIdentifier,
    StudioClient,
    StudioDataset,
    StudioExample,
)
from intelligence_layer.evaluation.aggregation.aggregator import AggregationLogic
from intelligence_layer.evaluation.benchmark.studio_benchmark import (
    create_aggregation_logic_identifier,
    create_evaluation_logic_identifier,
)
from intelligence_layer.evaluation.dataset.domain import Example
from intelligence_layer.evaluation.evaluation.evaluator.evaluator import (
    SingleOutputEvaluationLogic,
)


class DummyEvaluation(BaseModel):
    result: str


class DummyAggregatedEvaluation(BaseModel):
    score: float


class DummyEvaluationLogic(
    SingleOutputEvaluationLogic[
        str,
        str,
        None,
        DummyEvaluation,
    ]
):
    def do_evaluate_single_output(
        self,
        example: Example[str, None],
        output: str,
    ) -> DummyEvaluation:
        return DummyEvaluation(result="success")


class DummyAggregation(BaseModel):
    num_evaluations: int


class DummyAggregationLogic(AggregationLogic[DummyEvaluation, DummyAggregation]):
    def aggregate(self, evaluations: Iterable[DummyEvaluation]) -> DummyAggregation:
        return DummyAggregation(num_evaluations=len(list(evaluations)))


@fixture
def studio_dataset(
    studio_client: StudioClient, examples: Sequence[StudioExample[str, str]]
) -> str:
    return studio_client.submit_dataset(StudioDataset(name="dataset_name"), examples)


@fixture
def evaluation_logic_identifier() -> EvaluationLogicIdentifier:
    return create_evaluation_logic_identifier(DummyEvaluationLogic())


@fixture
def aggregation_logic_identifier() -> AggregationLogicIdentifier:
    return create_aggregation_logic_identifier(DummyAggregationLogic())


def test_create_benchmark(
    studio_client: StudioClient,
    studio_dataset: str,
    evaluation_logic_identifier: EvaluationLogicIdentifier,
    aggregation_logic_identifier: AggregationLogicIdentifier,
) -> None:
    benchmark_id = studio_client.create_benchmark(
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
    with pytest.raises(HTTPError, match=str(HTTPStatus.BAD_REQUEST.value)):
        studio_client.create_benchmark(
            "fake_id",
            evaluation_logic_identifier,
            aggregation_logic_identifier,
            "benchmark_name",
        )


def test_get_benchmark(
    studio_client: StudioClient,
    studio_dataset: str,
    evaluation_logic_identifier: EvaluationLogicIdentifier,
    aggregation_logic_identifier: AggregationLogicIdentifier,
) -> None:
    dummy_evaluation_logic = """return DummyEvaluation(result="success")"""
    benchmark_name = "benchmark_name"

    benchmark_id = studio_client.create_benchmark(
        studio_dataset,
        evaluation_logic_identifier,
        aggregation_logic_identifier,
        benchmark_name,
    )

    benchmark = studio_client.get_benchmark(benchmark_id)
    assert benchmark
    assert benchmark.dataset_id == studio_dataset
    assert dummy_evaluation_logic in benchmark.evaluation_logic.logic
    assert benchmark.project_id == studio_client.project_id
    assert benchmark.name == benchmark_name


def test_get_non_existing_benchmark(studio_client: StudioClient) -> None:
    benchmark = studio_client.get_benchmark(str(uuid4()))
    assert not benchmark
