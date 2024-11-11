from abc import ABC, abstractmethod
from typing import Any, Generic, Optional

from intelligence_layer.core import Input, Output
from intelligence_layer.core.task import Task
from intelligence_layer.evaluation.aggregation.aggregator import AggregationLogic
from intelligence_layer.evaluation.aggregation.domain import AggregatedEvaluation
from intelligence_layer.evaluation.dataset.domain import ExpectedOutput
from intelligence_layer.evaluation.evaluation.domain import Evaluation
from intelligence_layer.evaluation.evaluation.evaluator.evaluator import EvaluationLogic


class Benchmark(
    Generic[Input, Output, ExpectedOutput, Evaluation, AggregatedEvaluation]
):
    def __init__(
        self,
        benchmark_id: str,
        dataset_id: str,
        eval_logic: EvaluationLogic[Input, Output, ExpectedOutput, Evaluation],
        aggregation_logic: AggregationLogic[Evaluation, AggregatedEvaluation],
        **kwargs: Any,
    ) -> None:
        pass

    def run(self, task: Task[Input, Output], metadata: dict[str, Any]) -> str:
        return ""


class BenchmarkRepository(
    Generic[Input, Output, ExpectedOutput, Evaluation, AggregatedEvaluation], ABC
):
    @abstractmethod
    def create_benchmark(
        self,
        dataset_id: str,
        eval_logic: EvaluationLogic[Input, Output, ExpectedOutput, Evaluation],
        aggregation_logic: AggregationLogic[Evaluation, AggregatedEvaluation],
        name: str,
        metadata: Optional[dict[str, Any]] = None,
        description: Optional[str] = None,
    ) -> Benchmark[Input, Output, ExpectedOutput, Evaluation, AggregatedEvaluation]:
        pass

    @abstractmethod
    def get_benchmark(
        self,
        benchmark_id: str,
        eval_logic: EvaluationLogic[Input, Output, ExpectedOutput, Evaluation],
        aggregation_logic: AggregationLogic[Evaluation, AggregatedEvaluation],
        force: bool = False,
    ) -> Benchmark[Input, Output, ExpectedOutput, Evaluation, AggregatedEvaluation]:
        pass
