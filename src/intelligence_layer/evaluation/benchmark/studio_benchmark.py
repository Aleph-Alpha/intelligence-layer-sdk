import inspect
from collections.abc import Sequence
from datetime import datetime
from http import HTTPStatus
from typing import Any, Optional

import requests
from pydantic import TypeAdapter
from tqdm import tqdm

from intelligence_layer.connectors.studio.studio import (
    AggregationLogicIdentifier,
    BenchmarkLineage,
    EvaluationLogicIdentifier,
    PostBenchmarkExecution,
    StudioClient,
)
from intelligence_layer.core import Input, Output
from intelligence_layer.core.task import Task
from intelligence_layer.evaluation.aggregation.aggregator import (
    AggregationLogic,
    Aggregator,
)
from intelligence_layer.evaluation.aggregation.domain import AggregatedEvaluation
from intelligence_layer.evaluation.aggregation.in_memory_aggregation_repository import (
    InMemoryAggregationRepository,
)
from intelligence_layer.evaluation.benchmark.benchmark import (
    Benchmark,
    BenchmarkRepository,
)
from intelligence_layer.evaluation.dataset.domain import ExpectedOutput
from intelligence_layer.evaluation.dataset.studio_dataset_repository import (
    StudioDatasetRepository,
)
from intelligence_layer.evaluation.evaluation.domain import Evaluation
from intelligence_layer.evaluation.evaluation.evaluator.evaluator import (
    EvaluationLogic,
    Evaluator,
)
from intelligence_layer.evaluation.evaluation.in_memory_evaluation_repository import (
    InMemoryEvaluationRepository,
)
from intelligence_layer.evaluation.infrastructure.repository_navigator import (
    EvaluationLineage,
)
from intelligence_layer.evaluation.run.in_memory_run_repository import (
    InMemoryRunRepository,
)
from intelligence_layer.evaluation.run.runner import Runner


class StudioBenchmark(Benchmark):
    def __init__(
        self,
        benchmark_id: str,
        name: str,
        dataset_id: str,
        eval_logic: EvaluationLogic[Input, Output, ExpectedOutput, Evaluation],
        aggregation_logic: AggregationLogic[Evaluation, AggregatedEvaluation],
        studio_client: StudioClient,
        **kwargs: Any,
    ):
        self.id = benchmark_id
        self.name = name
        self.dataset_id = dataset_id
        self.eval_logic = eval_logic
        self.aggregation_logic = aggregation_logic
        self.client = studio_client
        self.run_repository = InMemoryRunRepository()
        self.evaluation_repository = InMemoryEvaluationRepository()
        self.aggregation_repository = InMemoryAggregationRepository()
        self.dataset_repository = StudioDatasetRepository(self.client)
        self.evaluator = Evaluator(
            self.dataset_repository,
            self.run_repository,
            self.evaluation_repository,
            f"benchmark-{self.id}-evaluator",
            self.eval_logic,
        )
        self.aggregator = Aggregator(
            self.evaluation_repository,
            self.aggregation_repository,
            f"benchmark-{self.id}-aggregator",
            self.aggregation_logic,
        )

    def execute(
        self,
        task: Task[Input, Output],
        name: str,
        description: Optional[str] = None,
        labels: Optional[set[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        start = datetime.now()

        runner = Runner(
            task,
            self.dataset_repository,
            self.run_repository,
            f"benchmark-{self.id}-runner",
        )
        run_overview = runner.run_dataset(
            self.dataset_id, description=description, labels=labels, metadata=metadata
        )

        evaluation_overview = self.evaluator.evaluate_runs(
            run_overview.id, description=description, labels=labels, metadata=metadata
        )

        aggregation_overview = self.aggregator.aggregate_evaluation(
            evaluation_overview.id,
            description=description,
            labels=labels,
            metadata=metadata,
        )

        end = datetime.now()

        data = PostBenchmarkExecution(
            name=name,
            description=description,
            labels=labels,
            metadata=metadata,
            start=start,
            end=end,
            run_start=run_overview.start,
            run_end=run_overview.end,
            run_successful_count=run_overview.successful_example_count,
            run_failed_count=run_overview.failed_example_count,
            run_success_avg_latency=0,  # TODO: Implement this
            run_success_avg_token_count=0,  # TODO: Implement this
            eval_start=evaluation_overview.start_date,
            eval_end=evaluation_overview.end_date,
            eval_successful_count=evaluation_overview.successful_evaluation_count,
            eval_failed_count=evaluation_overview.failed_evaluation_count,
            aggregation_start=aggregation_overview.start,
            aggregation_end=aggregation_overview.end,
            statistics=aggregation_overview.statistics.model_dump_json(),
        )

        benchmark_execution_id = self.client.create_benchmark_execution(
            benchmark_id=self.id, data=data
        )

        evaluation_lineages = list(
            self.evaluator.evaluation_lineages(evaluation_overview.id)
        )
        trace_ids = []
        for lineage in tqdm(evaluation_lineages, desc="Submitting traces to Studio"):
            trace = lineage.tracers[0]
            assert trace
            trace_id = self.client.submit_trace(trace.export_for_viewing())
            trace_ids.append(trace_id)

        benchmark_lineages = self._create_benchmark_lineages(
            eval_lineages=evaluation_lineages,
            trace_ids=trace_ids,
        )
        self.client.submit_benchmark_lineages(
            benchmark_lineages=benchmark_lineages,
            execution_id=benchmark_execution_id,
            benchmark_id=self.id,
        )

        return benchmark_execution_id

    def _create_benchmark_lineages(
        self,
        eval_lineages: list[
            EvaluationLineage[Input, ExpectedOutput, Output, Evaluation]
        ],
        trace_ids: list[str],
    ) -> Sequence[BenchmarkLineage]:
        return [
            self._create_benchmark_lineage(eval_lineage, trace_id)
            for eval_lineage, trace_id in zip(eval_lineages, trace_ids, strict=True)
        ]

    def _create_benchmark_lineage(
        self,
        eval_lineage: EvaluationLineage[Input, ExpectedOutput, Output, Evaluation],
        trace_id: str,
    ) -> BenchmarkLineage:
        return BenchmarkLineage(
            trace_id=trace_id,
            input=eval_lineage.example.input,
            expected_output=eval_lineage.example.expected_output,
            example_metadata=eval_lineage.example.metadata,
            output=eval_lineage.outputs[0].output,
            evaluation=eval_lineage.evaluation.result,
            run_latency=0,  # TODO: Implement this
            run_tokens=0,  # TODO: Implement this
        )


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
            benchmark_id = self.client.submit_benchmark(
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
            name,
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
        allow_diff: bool = False,
    ) -> StudioBenchmark | None:
        benchmark = self.client.get_benchmark(benchmark_id)
        if benchmark is None:
            return None
        return StudioBenchmark(
            benchmark_id,
            benchmark.name,
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
