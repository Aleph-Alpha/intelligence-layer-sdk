from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import (
    Callable,
    Generic,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    TypeVar,
    cast,
    final,
)
from uuid import uuid4

from tqdm import tqdm

from intelligence_layer.core.evaluation.domain import (
    AggregatedEvaluation,
    Dataset,
    Evaluation,
    EvaluationException,
    EvaluationRunOverview,
    Example,
    ExampleResult,
    ExpectedOutput,
    TaskSpanTrace,
)
from intelligence_layer.core.task import Input, Output, Task
from intelligence_layer.core.tracer import (
    InMemoryTaskSpan,
    InMemoryTracer,
    JsonSerializer,
    Tracer,
)


class EvaluationRepository(ABC):
    """Base evaluation repository interface.

    Provides methods to store and load evaluation results for individual examples
    of a run and the aggregated evaluation of said run.
    """

    @abstractmethod
    def evaluation_run_results(
        self, run_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleResult[Evaluation]]:
        """Returns all :class:`ExampleResult` instances of a given run

        Args:
            run_id: Identifier of the run to obtain the results for.
            evaluation_type: Type of evaluations that the :class:`Evaluator` returned
                in :func:`Evaluator.do_evaluate`

        Returns:
            All :class:`ExampleResult` of the run. Will return an empty list if there's none.
        """
        ...

    @abstractmethod
    def evaluation_example_result(
        self, run_id: str, example_id: str, evaluation_type: type[Evaluation]
    ) -> Optional[ExampleResult[Evaluation]]:
        """Returns an :class:`ExampleResult` of a given run by its id.

        Args:
            run_id: Identifier of the run to obtain the results for.
            example_id: Identifier of the :class:`ExampleResult` to be retrieved.
            evaluation_type: Type of evaluations that the `Evaluator` returned
                in :func:`Evaluator.do_evaluate`

        Returns:
            :class:`ExampleResult` if one was found, `None` otherwise.
        """
        ...

    @abstractmethod
    def store_example_result(
        self, run_id: str, result: ExampleResult[Evaluation]
    ) -> None:
        """Stores an :class:`ExampleResult` for a run in the repository.

        Args:
            run_id: Identifier of the run.
            result: The result to be persisted.
        """
        ...

    @abstractmethod
    def evaluation_run_overview(
        self, run_id: str, aggregation_type: type[AggregatedEvaluation]
    ) -> Optional[EvaluationRunOverview[AggregatedEvaluation]]:
        """Returns an :class:`EvaluationRunOverview` of a given run by its id.

        Args:
            run_id: Identifier of the run to obtain the overview for.
            aggregation_type: Type of aggregations that the :class:`Evaluator` returned
                in :func:`Evaluator.aggregate`

        Returns:
            :class:`EvaluationRunOverview` if one was found, `None` otherwise.
        """
        ...

    @abstractmethod
    def store_evaluation_run_overview(
        self, overview: EvaluationRunOverview[AggregatedEvaluation]
    ) -> None:
        """Stores an :class:`EvaluationRunOverview` in the repository.

        Args:
            overview: The overview to be persisted.
        """
        ...


T = TypeVar("T")


class CountingFilterIterable(Iterable[T]):
    def __init__(
        self, wrapped_iterable: Iterable[T], filter: Callable[[T], bool]
    ) -> None:
        self._wrapped_iterator = iter(wrapped_iterable)
        self._filter = filter
        self._included_count = 0
        self._excluded_count = 0

    def __next__(self) -> T:
        e = next(self._wrapped_iterator)
        while not self._filter(e):
            self._excluded_count += 1
            e = next(self._wrapped_iterator)
        self._included_count += 1
        return e

    def __iter__(self) -> Iterator[T]:
        return self

    def included_count(self) -> int:
        return self._included_count

    def excluded_count(self) -> int:
        return self._excluded_count


class Evaluator(
    ABC, Generic[Input, Output, ExpectedOutput, Evaluation, AggregatedEvaluation]
):
    """Base evaluator interface. This should run certain evaluation steps for some job.

    We suggest supplying a :class:`Task` in the `__init__` method and running it in the :func:`Evaluator.evaluate` method.

    Generics:
        Input: Interface to be passed to the :class:`Task` that shall be evaluated.
        Output: Type of the output of the :class:`Task` to be evaluated.
        ExpectedOutput: Output that is expected from the run with the supplied input.
        Evaluation: Interface of the metrics that come from the evaluated :class:`Task`.
        AggregatedEvaluation: The aggregated results of an evaluation run with a :class:`Dataset`.
    """

    def __init__(
        self, task: Task[Input, Output], repository: EvaluationRepository
    ) -> None:
        self.task = task
        self.repository = repository

    @abstractmethod
    def do_evaluate(
        self,
        input: Input,
        output: Output,
        expected_output: ExpectedOutput,
    ) -> Evaluation:
        """Executes the evaluation for this use-case.

        The implementation of this method is responsible for running a :class:`Task` (usually
        supplied by the __init__ method) and making any comparisons relevant to the evaluation.
        Based on the results, it should create an `Evaluation` class with all the metrics and return it.

        Args:
            input: The input that was passed to the :class:`Task` to produce the output
            output: Output of the :class:`Task` that shall be evaluated
            expected_output: Output that is compared to the generated output.

        Returns:
            The metrics that come from the evaluated :class:`Task`.
        """
        pass

    def _evaluate_example(
        self,
        run_id: str,
        example: Example[Input, ExpectedOutput],
        tracer: Tracer,
    ) -> Evaluation | EvaluationException:
        eval_tracer = InMemoryTracer()
        result = self.evaluate(example.input, example.expected_output, eval_tracer)
        example_result = ExampleResult(
            example_id=example.id,
            result=result,
            trace=TaskSpanTrace.from_task_span(
                cast(InMemoryTaskSpan, eval_tracer.entries[0])
            ),
        )
        self.repository.store_example_result(run_id=run_id, result=example_result)
        return example_result.result

    @final
    def evaluate(
        self, input: Input, expected_output: ExpectedOutput, tracer: Tracer
    ) -> Evaluation | EvaluationException:
        """Evaluates a single example and returns an `Evaluation` or `EvaluationException`.

        This will call the `run` method for the :class:`Task` defined in the `__init__` method.
        Catches any errors that occur during :func:`Task.run` or :func:`Evaluator.do_evaluate`.

        Args:
            input: Input for the :class:`Task`. Has to be same type as the input for the task used.
            expected_output: The expected output for the run.
                This will be used by the evaluator to compare the received output with.
            tracer: :class:`Tracer` used for tracing.
        Returns:
            Result of the evaluation or exception in case an error during running
            the :class:`Task` or evaluation.
        """

        try:
            output = self.task.run(input, tracer)
            return self.do_evaluate(input, output, expected_output)
        except Exception as e:
            return EvaluationException(error_message=str(e))

    @final
    def evaluate_dataset(
        self, dataset: Dataset[Input, ExpectedOutput], tracer: Tracer
    ) -> EvaluationRunOverview[AggregatedEvaluation]:
        """Evaluates an entire :class:`Dataset` in a threaded manner and aggregates the results into an `AggregatedEvaluation`.

        This will call the `run` method for each example in the :class:`Dataset`.
        Finally, it will call the `aggregate` method and return the aggregated results.

        Args:
            dataset: Dataset that will be used to evaluate a :class:`Task`.
            tracer: tracer used for tracing.
        Returns:
            The aggregated results of an evaluation run with a dataset.
        """
        run_id = str(uuid4())
        start = datetime.utcnow()
        with ThreadPoolExecutor(max_workers=10) as executor:
            evaluations = tqdm(
                executor.map(
                    lambda example: self._evaluate_example(
                        run_id,
                        example,
                        tracer,
                    ),
                    dataset.examples,
                ),
                desc="Evaluating",
            )
        # collect errors with debug log
        successful_evaluations = CountingFilterIterable(
            evaluations,
            lambda evaluation: not isinstance(evaluation, EvaluationException),
        )
        # The filter above ensures that only `Evaluation` instances are passed along
        statistics = self.aggregate(cast(Iterable[Evaluation], successful_evaluations))
        end = datetime.utcnow()

        run_overview = EvaluationRunOverview(
            id=run_id,
            statistics=statistics,
            start=start,
            end=end,
            dataset_name=dataset.name,
            successful_evaluation_count=successful_evaluations.included_count(),
            failed_evaluation_count=successful_evaluations.excluded_count(),
        )
        self.repository.store_evaluation_run_overview(run_overview)
        return run_overview

    @abstractmethod
    def aggregate(self, evaluations: Iterable[Evaluation]) -> AggregatedEvaluation:
        """`Evaluator`-specific method for aggregating individual `Evaluations` into report-like `Aggregated Evaluation`.

        This method is responsible for taking the results of an evaluation run and aggregating all the results.
        It should create an `AggregatedEvaluation` class and return it at the end.

        Args:
            evalautions: The results from running `evaluate_dataset` with a :class:`Task`.
        Returns:
            The aggregated results of an evaluation run with a :class:`Dataset`.
        """
        pass
