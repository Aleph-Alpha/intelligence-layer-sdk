import inspect
from http import HTTPStatus
from typing import Any, Optional

import requests
from pydantic import TypeAdapter

from intelligence_layer.connectors.studio.studio import (
    AggregationLogicIdentifier,
    EvaluationLogicIdentifier,
    StudioClient,
)
from intelligence_layer.core import Input, Output
from intelligence_layer.core.task import Task
from intelligence_layer.evaluation.aggregation.aggregator import (
    AggregationLogic,
    Aggregator,
)
from intelligence_layer.evaluation.aggregation.domain import AggregatedEvaluation
from intelligence_layer.evaluation.benchmark.benchmark import (
    Benchmark,
    BenchmarkRepository,
)
from intelligence_layer.evaluation.dataset.domain import ExpectedOutput
from intelligence_layer.evaluation.evaluation.domain import Evaluation
from intelligence_layer.evaluation.evaluation.evaluator.evaluator import (
    EvaluationLogic,
    Evaluator,
)


class StudioBenchmark(Benchmark):
    def __init__(
        self,
        benchmark_id: str,
        dataset_id: str,
        eval_logic: EvaluationLogic[Input, Output, ExpectedOutput, Evaluation],
        aggregation_logic: AggregationLogic[Evaluation, AggregatedEvaluation],
        studio_client: StudioClient,
        **kwargs: Any,
    ):
        self.id = benchmark_id
        self.dataset_id = dataset_id
        self.eval_logic = eval_logic
        self.aggregation_logic = aggregation_logic
        self.client = studio_client

    def run(self, task: Task[Input, Output], metadata: dict[str, Any]) -> str:
        raise NotImplementedError  # <- skip the impl here for now, not this is another ticket


class StudioBenchmarkRepository(BenchmarkRepository):
    def __init__(self, studio_client: StudioClient):
        self.client = studio_client

    def create_benchmark(
        self,
        dataset_id: str,
        eval_logic: EvaluationLogic[Input, Output, ExpectedOutput, Evaluation],
        aggregation_logic: AggregationLogic[Evaluation, AggregatedEvaluation],
        name: str,
        metadata: Optional[dict[str, Any]] = None,
        description: Optional[str] = None,
    ) -> StudioBenchmark:
        try:
            benchmark_id = self.client.create_benchmark(
                dataset_id,
                create_evaluation_logic_identifier(eval_logic),
                create_aggregation_logic_identifier(aggregation_logic),
                name,
                description,
                metadata,
            )
        except requests.HTTPError as e:
            if e.response.status_code == HTTPStatus.BAD_REQUEST:
                raise ValueError(f"Dataset with ID {dataset_id} not found") from e

        return StudioBenchmark(
            benchmark_id,
            dataset_id,
            eval_logic,
            aggregation_logic,
            studio_client=self.client,
        )

    def get_benchmark(
        self,
        benchmark_id: str,
        eval_logic: EvaluationLogic[Input, Output, ExpectedOutput, Evaluation],
        aggregation_logic: AggregationLogic[Evaluation, AggregatedEvaluation],
        force_execution: bool = False,
    ) -> StudioBenchmark:
        benchmark = self.client.get_benchmark(benchmark_id)
        if benchmark is None:
            raise ValueError("Benchmark not found")
        # check if the logic is the same
        # check force bool
        return StudioBenchmark(
            benchmark_id,
            benchmark.dataset_id,
            eval_logic,
            aggregation_logic,
            self.client,
        )


def type_to_schema(type_: type) -> dict[str, Any]:
    return TypeAdapter(type_).json_schema()


def create_evaluation_logic_identifier(
    eval_logic: EvaluationLogic[Input, Output, ExpectedOutput, Evaluation],
) -> EvaluationLogicIdentifier:
    evaluator = Evaluator(
        dataset_repository=None,  # type: ignore
        run_repository=None,  # type: ignore
        evaluation_repository=None,  # type: ignore
        description="",
        evaluation_logic=eval_logic,
    )
    return EvaluationLogicIdentifier(
        logic=inspect.getsource(type(eval_logic)),
        input_schema=type_to_schema(evaluator.input_type()),
        output_schema=type_to_schema(evaluator.output_type()),
        expected_output_schema=type_to_schema(evaluator.expected_output_type()),
        evaluation_schema=type_to_schema(evaluator.evaluation_type()),
    )


def create_aggregation_logic_identifier(
    aggregation_logic: AggregationLogic[Evaluation, AggregatedEvaluation],
) -> AggregationLogicIdentifier:
    aggregator = Aggregator(
        evaluation_repository=None,  # type: ignore
        aggregation_repository=None,  # type: ignore
        description="",
        aggregation_logic=aggregation_logic,
    )
    return AggregationLogicIdentifier(
        logic=inspect.getsource(type(aggregation_logic)),
        evaluation_schema=type_to_schema(aggregator.evaluation_type()),
        aggregation_schema=type_to_schema(aggregator.aggregated_evaluation_type()),
    )
