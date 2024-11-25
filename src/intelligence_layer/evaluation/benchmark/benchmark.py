from abc import ABC, abstractmethod
from typing import Any, Optional

from intelligence_layer.core import Input, Output
from intelligence_layer.core.task import Task
from intelligence_layer.evaluation.aggregation.aggregator import AggregationLogic
from intelligence_layer.evaluation.aggregation.domain import AggregatedEvaluation
from intelligence_layer.evaluation.dataset.domain import ExpectedOutput
from intelligence_layer.evaluation.evaluation.domain import Evaluation
from intelligence_layer.evaluation.evaluation.evaluator.evaluator import EvaluationLogic


class Benchmark(ABC):
    """Specific Benchmark instance used to run benchmark executions.

    Attributes:
        benchmark_id: Unique identifier for the benchmark.
        dataset_id: Identifier for the dataset used in the benchmark.
        eval_logic: Evaluation logic to be applied to task within the benchmark.
        aggregation_logic: Aggregation logic to combine individual evaluations into an aggregated result.
    """

    @abstractmethod
    def __init__(
        self,
        benchmark_id: str,
        dataset_id: str,
        eval_logic: EvaluationLogic[Input, Output, ExpectedOutput, Evaluation],
        aggregation_logic: AggregationLogic[Evaluation, AggregatedEvaluation],
        **kwargs: Any,
    ) -> None:
        """Initializes a Benchmark instance.

        Args:
            benchmark_id: Unique identifier for the benchmark.
            dataset_id: Identifier for the dataset associated with the benchmark.
            eval_logic: The evaluation logic to be applied.
            aggregation_logic: The aggregation logic to combine individual evaluations.
            **kwargs: Additional keyword arguments for extended configurations.
        """
        pass

    @abstractmethod
    def execute(
        self,
        task: Task[Input, Output],
        name: str,
        description: Optional[str],
        labels: Optional[set[str]],
        metadata: Optional[dict[str, Any]],
    ) -> str:
        """Executes the benchmark on a given task.

        Args:
            task: The task to be evaluated in the benchmark.
            name: Name of the benchmark execution.
            description: Description of the task to be evaluated.
            labels: Labels for filtering or categorizing the benchmark.
            metadata: Additional information about the task for logging or configuration.

        Returns:
            Identifier of the benchmark run.
        """
        pass


class BenchmarkRepository(ABC):
    """Used to manage Benchmark instances."""

    @abstractmethod
    def create_benchmark(
        self,
        dataset_id: str,
        eval_logic: EvaluationLogic[Input, Output, ExpectedOutput, Evaluation],
        aggregation_logic: AggregationLogic[Evaluation, AggregatedEvaluation],
        name: str,
        metadata: Optional[dict[str, Any]] = None,
        description: Optional[str] = None,
    ) -> Benchmark:
        """Creates a new benchmark and stores it in the repository.

        Args:
            dataset_id: Identifier for the dataset associated with the benchmark.
            eval_logic: Evaluation logic to be applied in the benchmark.
            aggregation_logic: Aggregation logic for combining individual evaluations.
            name: Name of the benchmark.
            metadata: Additional information about the benchmark, defaults to None.
            description: Description of the benchmark, defaults to None.

        Returns:
            The created benchmark instance.
        """
        pass

    @abstractmethod
    def get_benchmark(
        self,
        benchmark_id: str,
        eval_logic: EvaluationLogic[Input, Output, ExpectedOutput, Evaluation],
        aggregation_logic: AggregationLogic[Evaluation, AggregatedEvaluation],
        allow_diff: bool = False,
    ) -> Benchmark | None:
        """Retrieves an existing benchmark from the repository.

        Args:
            benchmark_id: Unique identifier for the benchmark to retrieve.
            eval_logic: Evaluation logic to apply.
            aggregation_logic: Aggregation logic to apply.
            allow_diff: Retrieve the benchmark even though logics behaviour do not match.

        Returns:
            The retrieved benchmark instance. Raises ValueError if no benchmark is found.
        """
        pass
