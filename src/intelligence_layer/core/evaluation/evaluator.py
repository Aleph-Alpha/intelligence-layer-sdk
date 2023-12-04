from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
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
    EvaluationOverview,
    EvaluationRunOverview,
    Example,
    ExampleEvaluation,
    ExampleOutput,
    ExampleTrace,
    ExpectedOutput,
    RunOverview,
)
from intelligence_layer.core.task import Input, Output, Task
from intelligence_layer.core.tracer import CompositeTracer, Tracer


class EvaluationRepository(ABC):
    """Base evaluation repository interface.

    Provides methods to store and load evaluation results for individual examples
    of a run and the aggregated evaluation of said run.
    """

    @abstractmethod
    def run_ids(self) -> Sequence[str]:
        """Returns the ids of all stored runs.

        Having the id of a run its outputs can be retrieved with
        :meth:`EvaluationRepository.example_outputs`.

        Returns:
            The ids of all stored runs.
        """
        ...

    @abstractmethod
    def eval_ids(self) -> Sequence[str]:
        """Returns the ids of all stored evaluation runs.

        Having the id of an evaluation run its overview can be retrieved with
        :meth:`EvaluationRepository.evaluation_run_overview`.

        Returns:
            The ids of all stored evaluation runs.
        """
        ...

    @abstractmethod
    def store_example_output(
        self, run_id: str, example_output: ExampleOutput[Output]
    ) -> None:
        ...

    @abstractmethod
    def example_outputs(
        self, run_id: str, output_type: type[Output]
    ) -> Iterable[ExampleOutput[Output]]:
        ...

    @abstractmethod
    def evaluation_run_results(
        self, run_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleEvaluation[Evaluation]]:
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
    def failed_evaluation_run_results(
        self, run_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleEvaluation[Evaluation]]:
        """Returns all failed :class:`ExampleResult` instances of a given run

        Args:
            run_id: Identifier of the run to obtain the results for.
            evaluation_type: Type of evaluations that the :class:`Evaluator` returned
                in :func:`Evaluator.do_evaluate`

        Returns:
            All failed :class:`ExampleResult` of the run. Will return an empty list if there's none.
        """
        ...

    @abstractmethod
    def evaluation_example_result(
        self, eval_id: str, example_id: str, evaluation_type: type[Evaluation]
    ) -> Optional[ExampleEvaluation[Evaluation]]:
        """Returns an :class:`ExampleResult` of a given run by its id.

        Args:
            eval_id: Identifier of the run to obtain the results for.
            example_id: Identifier of the :class:`ExampleResult` to be retrieved.
            evaluation_type: Type of evaluations that the `Evaluator` returned
                in :func:`Evaluator.do_evaluate`

        Returns:
            :class:`ExampleResult` if one was found, `None` otherwise.
        """
        ...

    @abstractmethod
    def evaluation_example_trace(
        self, run_id: str, example_id: str
    ) -> Optional[ExampleTrace]:
        ...

    @abstractmethod
    def example_tracer(self, run_id: str, example_id: str) -> Tracer:
        ...

    @abstractmethod
    def store_example_result(
        self, run_id: str, result: ExampleEvaluation[Evaluation]
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

    Arguments:
        task: The task that will be evaluated.
        repository: The repository that will be used to store evaluation results.
        directory: If specified, evaluation traces will be stored here to be used by the trace-viewer.

    Generics:
        Input: Interface to be passed to the :class:`Task` that shall be evaluated.
        Output: Type of the output of the :class:`Task` to be evaluated.
        ExpectedOutput: Output that is expected from the run with the supplied input.
        Evaluation: Interface of the metrics that come from the evaluated :class:`Task`.
        AggregatedEvaluation: The aggregated results of an evaluation run with a :class:`Dataset`.
    """

    def __init__(
        self,
        task: Task[Input, Output],
        repository: EvaluationRepository,
        directory: Optional[Path] = None,
    ) -> None:
        self._task = task
        self._repository = repository
        self._directory = directory

    @abstractmethod
    def output_type(self) -> type[Output]:
        ...

    @abstractmethod
    def evaluation_type(self) -> type[Evaluation]:
        ...

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

    @final
    def _evaluate_example(
        self,
        run_id: str,
        example: Example[Input, ExpectedOutput],
        tracer: Optional[Tracer] = None,
    ) -> Evaluation | EvaluationException:
        evaluate_tracer = (
            CompositeTracer(
                tracers=[tracer, self._repository.example_tracer(run_id, example.id)]
            )
            if tracer
            else self._repository.example_tracer(run_id, example.id)
        )
        result = self.evaluate(example.input, example.expected_output, evaluate_tracer)
        example_result = ExampleEvaluation(
            example_id=example.id,
            result=result,
        )
        self._repository.store_example_result(run_id=run_id, result=example_result)
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
            output = self._task.run(input, tracer)
            return self.do_evaluate(input, output, expected_output)
        except Exception as e:
            return EvaluationException(error_message=str(e))

    def run_dataset(
        self, dataset: Dataset[Input, ExpectedOutput], tracer: Optional[Tracer] = None
    ) -> RunOverview:
        def run(example: Example[Input, ExpectedOutput]) -> None:
            evaluate_tracer = self._repository.example_tracer(run_id, example.id)
            if tracer:
                evaluate_tracer = CompositeTracer([evaluate_tracer, tracer])
            output = self._task.run(example.input, evaluate_tracer)
            self._repository.store_example_output(
                run_id, ExampleOutput(example_id=example.id, output=output)
            )

        run_id = str(uuid4())
        start = datetime.utcnow()
        with ThreadPoolExecutor(max_workers=10) as executor:
            tqdm(executor.map(run, dataset.examples), desc="Evaluating")
        return RunOverview(
            dataset_name=dataset.name,
            id=run_id,
            start=start,
            end=datetime.utcnow(),
            failed_example_count=0,
            successful_example_count=0,
        )

    def evaluate_run(
        self, dataset: Dataset[Input, ExpectedOutput], run_overview: RunOverview
    ) -> EvaluationOverview:
        eval_id = str(uuid4())
        start = datetime.utcnow()
        successful_count = 0
        for example_output in self._repository.example_outputs(
            run_overview.id, self.output_type()
        ):
            example = dataset.example(example_output.example_id)
            assert example
            # TODO this will eventually produce a side-effect as in case of human eval
            #  the result will not be available (but maybe next step)
            evaluation = self.do_evaluate(
                example.input, example_output.output, example.expected_output
            )
            self._repository.store_example_result(
                eval_id, ExampleEvaluation(example_id=example.id, result=evaluation)
            )
            successful_count += 1
        return EvaluationOverview(
            run_overview=run_overview,
            id=eval_id,
            start=start,
            end=datetime.utcnow(),
            failed_evaluation_count=0,
            successful_evaluation_count=successful_count,
        )

    def aggregate_evaluation(
        self, evaluation_overview: EvaluationOverview
    ) -> EvaluationRunOverview[AggregatedEvaluation]:
        example_results = self._repository.evaluation_run_results(
            evaluation_overview.id, self.evaluation_type()
        )

        statistics = self.aggregate(
            (
                example_result.result
                for example_result in example_results
                if not isinstance(example_result.result, EvaluationException)
            )
        )

        run_overview = EvaluationRunOverview(
            evaluation_overview=evaluation_overview,
            statistics=statistics,
        )
        self._repository.store_evaluation_run_overview(run_overview)
        return run_overview

    @final
    def evaluate_dataset(
        self,
        dataset: Dataset[Input, ExpectedOutput],
        tracer: Optional[Tracer] = None,
    ) -> EvaluationRunOverview[AggregatedEvaluation]:
        """Evaluates an entire :class:`Dataset` in a threaded manner and aggregates the results into an `AggregatedEvaluation`.

        This will call the `run` method for each example in the :class:`Dataset`.
        Finally, it will call the `aggregate` method and return the aggregated results.

        Args:
            dataset: Dataset that will be used to evaluate a :class:`Task`.
            tracer: Optional tracer used for extra tracing.
                Traces are always saved in the evaluation repository.

        Returns:
            The aggregated results of an evaluation run with a dataset.
        """
        run_id = self.run_dataset(dataset, tracer)
        eval_id = self.evaluate_run(dataset, run_id)
        return self.aggregate_evaluation(eval_id)
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
        self._repository.store_evaluation_run_overview(run_overview)
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
