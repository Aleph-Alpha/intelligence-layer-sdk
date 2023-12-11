from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from inspect import get_annotations
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

from intelligence_layer.connectors import ArgillaClient, Field
from intelligence_layer.connectors.argilla.argilla_client import (
    ArgillaEvaluation,
    Question,
    RecordData,
)
from intelligence_layer.core.evaluation.domain import (
    AggregatedEvaluation,
    Dataset,
    Evaluation,
    EvaluationOverview,
    Example,
    ExampleEvaluation,
    ExampleOutput,
    ExampleTrace,
    ExpectedOutput,
    FailedExampleEvaluation,
    FailedExampleRun,
    PartialEvaluationOverview,
    RunOverview,
)
from intelligence_layer.core.task import Input, Output, Task
from intelligence_layer.core.tracer import CompositeTracer, Tracer

EvaluationOverviewType = TypeVar(
    "EvaluationOverviewType", bound=PartialEvaluationOverview
)


class EvaluationRepository(ABC):
    """Base evaluation repository interface.

    Provides methods to store and load evaluation results for individual examples
    of a run and the aggregated evaluation of said run.
    """

    @abstractmethod
    def run_ids(self) -> Sequence[str]:
        """Returns the ids of all stored runs.

        Having the id of a run, its outputs can be retrieved with
        :meth:`EvaluationRepository.example_outputs`.

        Returns:
            The ids of all stored runs.
        """
        ...

    @abstractmethod
    def eval_ids(self) -> Sequence[str]:
        """Returns the ids of all stored evaluation runs.

        Having the id of an evaluation run, its overview can be retrieved with
        :meth:`EvaluationRepository.evaluation_run_overview`.

        Returns:
            The ids of all stored evaluation runs.
        """
        ...

    @abstractmethod
    def example_outputs(
        self, run_id: str, output_type: type[Output]
    ) -> Iterable[ExampleOutput[Output]]:
        """Returns all :class:`ExampleOutput` for a given run.

        Args:
            run_id: The unique identifier of the run.
            output_type: Type of output that the `Task` returned
                in :func:`Task.do_run`

        Returns:
            Iterable over all outputs.
        """
        ...

    @abstractmethod
    def store_example_output(
        self, run_id: str, example_output: ExampleOutput[Output]
    ) -> None:
        """Stores an individual :class:`ExampleOutput` for a given run.

        Args:
            run_id: The unique identifier of the run.
            example_output: The actual output.
        """
        ...

    @abstractmethod
    def example_trace(self, run_id: str, example_id: str) -> Optional[ExampleTrace]:
        """Returns an :class:`ExampleTrace` for an example in a run.

        Args:
            run_id: The unique identifier of the run.
            example_id: Example identifier, will match :class:`ExampleEvaluation` identifier.
            example_output: The actual output.
        """
        ...

    @abstractmethod
    def example_tracer(self, run_id: str, example_id: str) -> Tracer:
        """Returns a :class:`Tracer` to trace an individual example run.

        Args:
            run_id: The unique identifier of the run.
            example_id: Example identifier, will match :class:`ExampleEvaluation` identifier.
        """
        ...

    @abstractmethod
    def example_evaluation(
        self, eval_id: str, example_id: str, evaluation_type: type[Evaluation]
    ) -> Optional[ExampleEvaluation[Evaluation]]:
        """Returns an :class:`ExampleEvaluation` of a given run by its id.

        Args:
            eval_id: Identifier of the run to obtain the results for.
            example_id: Example identifier, will match :class:`ExampleEvaluation` identifier.
            evaluation_type: Type of evaluations that the `Evaluator` returned
                in :func:`Evaluator.do_evaluate`

        Returns:
            :class:`ExampleEvaluation` if one was found, `None` otherwise.
        """
        ...

    @abstractmethod
    def store_example_evaluation(
        self, eval_id: str, result: ExampleEvaluation[Evaluation]
    ) -> None:
        """Stores an :class:`ExampleEvaluation` for a run in the repository.

        Args:
            eval_id: Identifier of the eval run.
            result: The result to be persisted.
        """
        ...

    @abstractmethod
    def example_evaluations(
        self, eval_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleEvaluation[Evaluation]]:
        """Returns all :class:`ExampleResult` instances of a given run

        Args:
            eval_id: Identifier of the eval run to obtain the results for.
            evaluation_type: Type of evaluations that the :class:`Evaluator` returned
                in :func:`Evaluator.do_evaluate`

        Returns:
            All :class:`ExampleResult` of the run. Will return an empty list if there's none.
        """
        ...

    @abstractmethod
    def failed_example_evaluations(
        self, eval_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleEvaluation[Evaluation]]:
        """Returns all failed :class:`ExampleResult` instances of a given run

        Args:
            eval_id: Identifier of the eval run to obtain the results for.
            evaluation_type: Type of evaluations that the :class:`Evaluator` returned
                in :func:`Evaluator.do_evaluate`

        Returns:
            All failed :class:`ExampleResult` of the run. Will return an empty list if there's none.
        """
        ...

    @abstractmethod
    def evaluation_overview(
        self, eval_id: str, overview_type: type[EvaluationOverviewType]
    ) -> EvaluationOverviewType | None:
        """Returns an :class:`EvaluationRunOverview` of a given run by its id.

        Args:
            eval_id: Identifier of the eval run to obtain the overview for.
            aggregation_type: Type of aggregations that the :class:`Evaluator` returned
                in :func:`Evaluator.aggregate`

        Returns:
            :class:`EvaluationRunOverview` if one was found, `None` otherwise.
        """
        ...

    @abstractmethod
    def store_evaluation_overview(self, overview: PartialEvaluationOverview) -> None:
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


class BaseEvaluator(
    ABC, Generic[Input, Output, ExpectedOutput, Evaluation, AggregatedEvaluation]
):
    """Base evaluator interface.

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
    ) -> None:
        self._task = task
        self._repository = repository

    def output_type(self) -> type[Output]:
        """Returns the type of the evaluated task's output.

        This can be used to retrieve properly typed outputs of an evaluation run
        from a :class:`EvaluationRepository`

        Returns:
            the type of the evaluated task's output.
        """
        output_type = get_annotations(self._task.do_run).get("return", None)
        if not output_type:
            raise TypeError(
                f"Task of type {type(self._task)} must have a type-hint for the return value of do_run to detect the output_type. "
                f"Alternatively overwrite output_type() in {type(self)}"
            )
        return cast(type[Output], output_type)

    @abstractmethod
    def evaluation_type(self) -> type[Evaluation]:
        """Returns the type of the evaluation result of an example.

        This can be used to retrieve properly typed evaluations of an evaluation run
        from a :class:`EvaluationRepository`

        Returns:
            Returns the type of the evaluation result of an example.
        """
        ...

    @abstractmethod
    def evaluate(
        self, example: Example[Input, ExpectedOutput], eval_id: str, output: Output
    ) -> None:
        ...

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
        ...

    def _create_dataset(self) -> str:
        """Generates an ID for the dataset and creates it if necessary.

        If no extra logic is required to create the dataset for the run,
        this function just returns a UUID as string.
        In other cases (like when the dataset has to be created in an external repository),
        this method is responsible for implementing this logic and returning the ID.

        Returns:
            The ID of the dataset for reference.
        """
        return str(uuid4())

    @final
    def run_dataset(
        self, dataset: Dataset[Input, ExpectedOutput], tracer: Optional[Tracer] = None
    ) -> RunOverview:
        """Generates all outputs for the provided dataset.

        Will run each :class:`Example` provided in the dataset through the :class:`Task`.

        Args:
            dataset: The :class:`Dataset` to generate output for. Consists of examples, each
                with an :class:`Input` and an :class:`ExpectedOutput` (can be None).
            output: Output of the :class:`Task` that shall be evaluated

        Returns:
            An overview of the run. Outputs will not be returned but instead stored in the
            :class:`EvaluationRepository` provided in the __init__.
        """

        def run(example: Example[Input, ExpectedOutput]) -> Output | FailedExampleRun:
            evaluate_tracer = self._repository.example_tracer(run_id, example.id)
            if tracer:
                evaluate_tracer = CompositeTracer([evaluate_tracer, tracer])
            try:
                return self._task.run(example.input, evaluate_tracer)
            except Exception as e:
                return FailedExampleRun.from_exception(e)

        run_id = str(uuid4())
        start = datetime.utcnow()
        with ThreadPoolExecutor(max_workers=10) as executor:
            outputs = tqdm(executor.map(run, dataset.examples), desc="Evaluating")

        failed_count = 0
        successful_count = 0
        for output, example in zip(outputs, dataset.examples):
            if isinstance(output, FailedExampleRun):
                failed_count += 1
            else:
                successful_count += 1
            self._repository.store_example_output(
                run_id, ExampleOutput[Output](example_id=example.id, output=output)
            )

        return RunOverview(
            dataset_name=dataset.name,
            id=run_id,
            start=start,
            end=datetime.utcnow(),
            failed_example_count=failed_count,
            successful_example_count=successful_count,
        )

    @final
    def evaluate_run(
        self, dataset: Dataset[Input, ExpectedOutput], run_overview: RunOverview
    ) -> PartialEvaluationOverview:
        """Evaluates all generated outputs in the run.

        For each example in the dataset & its corresponding (generated) output,
        :func:`BaseEvaluator.do_evaluate` is called and eval metrics are produced &
        stored in the provided :class:`EvaluationRepository`.

        Args:
            dataset: The :class:`Dataset` the outputs were generated for. Consists of
                examples, each with an :class:`Input` and an :class:`ExpectedOutput`
                (can be None).
            run_overview: An overview of the run to be evaluated. Does not include
                outputs as these will be retrieved from the repository.

        Returns:
            An overview of the evaluation. Individual :class:`Evaluation`s will not be
            returned but instead stored in the :class:`EvaluationRepository` provided in the
            __init__.
        """

        eval_id = self._create_dataset()
        start = datetime.utcnow()
        successful_output_iter = (
            output
            for output in self._repository.example_outputs(
                run_overview.id, self.output_type()
            )
            if not isinstance(output.output, FailedExampleRun)
        )
        for example_output in successful_output_iter:
            example = dataset.example(example_output.example_id)
            assert example
            assert not isinstance(example_output.output, FailedExampleRun)
            self.evaluate(example, eval_id, example_output.output)

        partial_overview = PartialEvaluationOverview(
            run_overview=run_overview, id=eval_id, start=start
        )
        self._repository.store_evaluation_overview(partial_overview)

        return partial_overview

    def aggregate_evaluation(
        self, eval_id: str
    ) -> EvaluationOverview[AggregatedEvaluation]:
        """Aggregates all evaluations into an overview that includes high-level statistics.

        Aggregates :class:`Evaluation`s according to the implementation of :func:`BaseEvaluator.aggregate`.

        Args:
            evaluation_overview: An overview of the evaluation to be aggregated. Does not include
                actual evaluations as these will be retrieved from the repository.

        Returns:
            An overview of the aggregated evaluation.
        """

        evaluation_overview = self._repository.evaluation_overview(
            eval_id, PartialEvaluationOverview
        )
        if not evaluation_overview:
            raise ValueError(
                f"No PartialEvaluationOverview found for eval-id: {eval_id}"
            )
        example_evaluations = self._repository.example_evaluations(
            evaluation_overview.id, self.evaluation_type()
        )
        successful_evaluations = CountingFilterIterable(
            (example_eval.result for example_eval in example_evaluations),
            lambda evaluation: not isinstance(evaluation, FailedExampleEvaluation),
        )
        statistics = self.aggregate(cast(Iterable[Evaluation], successful_evaluations))
        run_overview = EvaluationOverview(
            statistics=statistics,
            end=datetime.utcnow(),
            successful_count=successful_evaluations.included_count(),
            failed_evaluation_count=successful_evaluations.excluded_count(),
            **(evaluation_overview.model_dump()),
        )
        self._repository.store_evaluation_overview(run_overview)
        return run_overview


class Evaluator(
    BaseEvaluator[Input, Output, ExpectedOutput, Evaluation, AggregatedEvaluation]
):
    """Evaluator that can handle automatic evaluation scenarios.

    This evaluator should be used for automatic eval. A user still has to implement
    :func:`BaseEvaluator.do_evaluate` and :func:`BaseEvaluator.aggregate`.

    Arguments:
        task: The task that will be evaluated.
        repository: The repository that will be used to store evaluation results.

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
    ) -> None:
        super().__init__(task, repository)

    def evaluation_type(self) -> type[Evaluation]:
        evaluation_type = get_annotations(self.do_evaluate).get("return", None)
        if not evaluation_type:
            raise TypeError(
                f"Evaluator of type {type(self)} must have a type-hint for the return value of do_evaluate to detect evaluation_type. "
                f"Alternatively overwrite its `evaluation_type()`"
            )
        return cast(type[Evaluation], evaluation_type)

    @abstractmethod
    def do_evaluate(
        self,
        input: Input,
        output: Output,
        expected_output: ExpectedOutput,
    ) -> Evaluation:
        """Executes the evaluation for this use-case.

        Responsible for comparing the input & expected output of a task to the
        actually generated output.

        Args:
            input: The input that was passed to the :class:`Task` to produce the output.
            output: Output of the :class:`Task` that shall be evaluated.
            expected_output: Output that is compared to the generated output.

        Returns:
            The metrics that come from the evaluated :class:`Task`.
        """
        pass

    @final
    def evaluate(
        self, example: Example[Input, ExpectedOutput], eval_id: str, output: Output
    ) -> None:
        try:
            result: Evaluation | FailedExampleEvaluation = self.do_evaluate(
                example.input, output, example.expected_output
            )
        except Exception as e:
            result = FailedExampleEvaluation.from_exception(e)
        self._repository.store_example_evaluation(
            eval_id, ExampleEvaluation(example_id=example.id, result=result)
        )

    @final
    def run_and_evaluate(
        self, input: Input, expected_output: ExpectedOutput, tracer: Tracer
    ) -> Evaluation | FailedExampleEvaluation:
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
            return FailedExampleEvaluation.from_exception(e)

    @final
    def evaluate_dataset(
        self,
        dataset: Dataset[Input, ExpectedOutput],
        tracer: Optional[Tracer] = None,
    ) -> EvaluationOverview[AggregatedEvaluation]:
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
        run_overview = self.run_dataset(dataset, tracer)
        partial_evaluation_overview = self.evaluate_run(dataset, run_overview)
        return self.aggregate_evaluation(partial_evaluation_overview.id)


class ArgillaEvaluator(
    BaseEvaluator[Input, Output, ExpectedOutput, Evaluation, AggregatedEvaluation], ABC
):
    def __init__(
        self,
        task: Task[Input, Output],
        repository: EvaluationRepository,
        argilla_client: ArgillaClient,
        workspace_id: str,
    ) -> None:
        super().__init__(task, repository)
        self._client = argilla_client
        self._workspace_id = workspace_id

    def evaluation_type(self) -> type[Evaluation]:
        evaluation_type = get_annotations(self.do_evaluate).get("return", None)
        if not evaluation_type:
            raise TypeError(
                f"Evaluator of type {type(self)} must have a type-hint for the return value of do_evaluate to detect evaluation_type. "
                f"Alternatively overwrite its `evaluation_type()`"
            )
        return cast(type[Evaluation], evaluation_type)

    @final
    def _create_dataset(self) -> str:
        return self._client.create_dataset(
            self._workspace_id,
            str(uuid4()),
            self._dataset_fields(),
            self._dataset_questions(),
        )

    @final
    def partial_evaluate_dataset(
        self,
        dataset: Dataset[Input, ExpectedOutput],
        tracer: Optional[Tracer] = None,
    ) -> PartialEvaluationOverview:
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
        run_overview = self.run_dataset(dataset, tracer)
        return self.evaluate_run(dataset, run_overview)

    @abstractmethod
    def _dataset_fields(self) -> Sequence[Field]:
        ...

    @abstractmethod
    def _dataset_questions(self) -> Sequence[Question]:
        ...

    @abstractmethod
    def _to_record(self, example_id: str, input: Input, output: Output) -> RecordData:
        ...

    @abstractmethod
    def aggregate(
        self,
        evaluations: Iterable[Evaluation],
        argilla_evaluations: Iterable[ArgillaEvaluation] = iter([]),
    ) -> AggregatedEvaluation:
        """`Evaluator`-specific method for aggregating individual `Evaluations` into report-like `Aggregated Evaluation`.

        This method is responsible for taking the results of an evaluation run and aggregating all the results.
        It should create an `AggregatedEvaluation` class and return it at the end.

        Args:
            evalautions: The results from running `evaluate_dataset` with a :class:`Task`.

        Returns:
            The aggregated results of an evaluation run with a :class:`Dataset`.
        """
        ...

    @abstractmethod
    def do_evaluate(
        self,
        input: Input,
        output: Output,
        expected_output: ExpectedOutput,
    ) -> Evaluation:
        """Executes the evaluation for this use-case.

        Responsible for comparing the input & expected output of a task to the
        actually generated output.

        Args:
            input: The input that was passed to the :class:`Task` to produce the output.
            output: Output of the :class:`Task` that shall be evaluated.
            expected_output: Output that is compared to the generated output.

        Returns:
            The metrics that come from the evaluated :class:`Task`.
        """
        pass

    def aggregate_evaluation(
        self, eval_id: str
    ) -> EvaluationOverview[AggregatedEvaluation]:
        """Aggregates all evaluations into an overview that includes high-level statistics.

        Aggregates :class:`Evaluation`s according to the implementation of :func:`BaseEvaluator.aggregate`.

        Args:
            evaluation_overview: An overview of the evaluation to be aggregated. Does not include
                actual evaluations as these will be retrieved from the repository.

        Returns:
            An overview of the aggregated evaluation.
        """
        evaluation_overview = self._repository.evaluation_overview(
            eval_id, PartialEvaluationOverview
        )
        if not evaluation_overview:
            raise ValueError(
                f"No PartialEvaluationOverview found for eval-id: {eval_id}"
            )
        example_evaluations = self._repository.example_evaluations(
            evaluation_overview.id, self.evaluation_type()
        )
        successful_evaluations = CountingFilterIterable(
            (example_eval.result for example_eval in example_evaluations),
            lambda evaluation: not isinstance(evaluation, FailedExampleEvaluation),
        )
        argilla_evaluations = self._client.evaluations(eval_id)
        statistics = self.aggregate(
            cast(Iterable[Evaluation], successful_evaluations), argilla_evaluations
        )
        run_overview = EvaluationOverview(
            statistics=statistics,
            end=datetime.utcnow(),
            successful_count=successful_evaluations.included_count(),
            failed_evaluation_count=successful_evaluations.excluded_count(),
            **(evaluation_overview.model_dump()),
        )
        self._repository.store_evaluation_overview(run_overview)
        return run_overview

    @final
    def evaluate(
        self, example: Example[Input, ExpectedOutput], eval_id: str, output: Output
    ) -> None:
        self._client.add_record(
            eval_id, self._to_record(example.id, example.input, output)
        )
        try:
            result: Evaluation | FailedExampleEvaluation = self.do_evaluate(
                example.input, output, example.expected_output
            )
        except Exception as e:
            result = FailedExampleEvaluation.from_exception(e)
        self._repository.store_example_evaluation(
            eval_id, ExampleEvaluation(example_id=example.id, result=result)
        )
