from abc import ABC, abstractmethod
from datetime import datetime
from functools import lru_cache
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
    final,
    get_args,
    get_origin,
)
from uuid import uuid4

from intelligence_layer.connectors import ArgillaClient, Field
from intelligence_layer.connectors.argilla.argilla_client import (
    ArgillaEvaluation,
    Question,
    RecordData,
)
from intelligence_layer.core.evaluation.domain import (
    AggregatedEvaluation,
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
from intelligence_layer.core.tracer import Tracer

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

    @abstractmethod
    def run_overview(self, run_id: str) -> RunOverview | None:
        """Returns an :class:`RunOverview` of a given run by its id.

        Args:
            run_id: Identifier of the eval run to obtain the overview for.

        Returns:
            :class:`RunOverview` if one was found, `None` otherwise.
        """
        ...

    @abstractmethod
    def store_run_overview(self, overview: RunOverview) -> None:
        """Stores an :class:`RunOverview` in the repository.

        Args:
            overview: The overview to be persisted.
        """
        ...


class DatasetRepository(ABC):
    @abstractmethod
    def create_dataset(
        self,
        examples: Iterable[Example[Input, ExpectedOutput]],
    ) -> str:
        ...

    @abstractmethod
    def examples_by_id(
        self,
        dataset_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
    ) -> Optional[Iterable[Example[Input, ExpectedOutput]]]:
        ...

    @abstractmethod
    def example(
        self,
        dataset_id: str,
        example_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
    ) -> Optional[Example[Input, ExpectedOutput]]:
        ...

    @abstractmethod
    def delete_dataset(self, dataset_id: str) -> None:
        ...

    @abstractmethod
    def list_datasets(self) -> Iterable[str]:
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
        evaluation_repository: The repository that will be used to store evaluation results.

    Generics:
        Input: Interface to be passed to the :class:`Task` that shall be evaluated.
        Output: Type of the output of the :class:`Task` to be evaluated.
        ExpectedOutput: Output that is expected from the run with the supplied input.
        Evaluation: Interface of the metrics that come from the evaluated :class:`Task`.
        AggregatedEvaluation: The aggregated results of an evaluation run with a dataset.
    """

    def __init__(
        self,
        evaluation_repository: EvaluationRepository,
        dataset_repository: DatasetRepository,
    ) -> None:
        self._evaluation_repository = evaluation_repository
        self._dataset_repository = dataset_repository

    @lru_cache(maxsize=1)
    def _get_types(self) -> Mapping[str, type]:  # noqa
        def is_not_type_var(object: Any) -> bool:
            return type(object) is not TypeVar

        def types_of_base(orig_bases: Any) -> list[Union[type, TypeVar]]:
            return list(get_args(orig_bases))

        def is_eligible_subclass(parent: type) -> bool:
            return hasattr(parent, "__orig_bases__") and issubclass(
                parent, BaseEvaluator
            )

        def update_types() -> None:
            num_types_set = 0
            for current_index, current_type in enumerate(current_types):
                if is_not_type_var(current_type):
                    type_var_count = num_types_set - 1
                    for element_index, element in enumerate(type_list):
                        if not is_not_type_var(element):
                            type_var_count += 1
                        if type_var_count == current_index:
                            break
                    assert type_var_count == current_index
                    type_list[element_index] = current_type
                    num_types_set += 1

        # mypy does not know __orig_bases__
        base_evaluator_bases = BaseEvaluator.__orig_bases__[1]  # type: ignore
        type_list: list[type | TypeVar] = types_of_base(base_evaluator_bases)
        for parent in (
            p for p in reversed(type(self).__mro__) if is_eligible_subclass(p)
        ):
            # mypy does not know __orig_bases__
            for base in parent.__orig_bases__:  # type: ignore
                origin = get_origin(base)
                if origin is None or not issubclass(origin, BaseEvaluator):
                    continue
                current_types = types_of_base(base)
                update_types()
        assert all(is_not_type_var(t) for t in type_list)
        return {
            name: param_type
            for name, param_type in zip(
                (a.__name__ for a in get_args(base_evaluator_bases)),
                cast(Sequence[type], type_list),
            )
        }

    def input_type(self) -> type[Input]:
        try:
            input_type = self._get_types()["Input"]
        except KeyError:
            raise TypeError(f"Alternatively overwrite input_type() in {type(self)}")
        return cast(type[Input], input_type)

    def output_type(self) -> type[Output]:
        """Returns the type of the evaluated task's output.

        This can be used to retrieve properly typed outputs of an evaluation run
        from a :class:`EvaluationRepository`

        Returns:
            the type of the evaluated task's output.
        """
        try:
            output_type = self._get_types()["Output"]
        except KeyError:
            raise TypeError(f"Alternatively overwrite output_type() in {type(self)}")
        return cast(type[Output], output_type)

    def expected_output_type(self) -> type[ExpectedOutput]:
        try:
            expected_output_type = self._get_types()["ExpectedOutput"]
        except KeyError:
            raise TypeError(
                f"Alternatively overwrite expected_output_type() in {type(self)}"
            )
        return cast(type[ExpectedOutput], expected_output_type)

    def evaluation_type(self) -> type[Evaluation]:
        """Returns the type of the evaluation result of an example.

        This can be used to retrieve properly typed evaluations of an evaluation run
        from a :class:`EvaluationRepository`

        Returns:
            Returns the type of the evaluation result of an example.
        """
        try:
            evaluation_type = self._get_types()["Evaluation"]
        except KeyError:
            raise TypeError(
                f"Alternatively overwrite evaluation_type() in {type(self)}"
            )
        return cast(type[Evaluation], evaluation_type)

    @abstractmethod
    def evaluate(
        self, example: Example[Input, ExpectedOutput], eval_id: str, *output: Output
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

    def _create_evaluation_dataset(self) -> str:
        """Generates an ID for the dataset and creates it if necessary.

        If no extra logic is required to create the dataset for the run,
        this function just returns a UUID as string.
        In other cases (like when the dataset has to be created in an external repository),
        this method is responsible for implementing this logic and returning the ID.

        Returns:
            The ID of the dataset used for retrieval.
        """
        return str(uuid4())

    @final
    def evaluate_runs(self, *run_ids: str) -> PartialEvaluationOverview:
        """Evaluates all generated outputs in the run.

        For each set of successful outputs in the referenced runs,
        :func:`BaseEvaluator.do_evaluate` is called and eval metrics are produced &
        stored in the provided :class:`EvaluationRepository`.

        Args:
            run_ids: The runs to be evaluated. Each run is expected to have the same
                dataset as input (which implies their tasks have the same input-type)
                and their tasks have the same output-type. For each example in the
                dataset referenced by the runs the outputs of all runs are collected
                and if all of them were successful they are passed on to the implementation
                specific evaluation. For a simple evaluation only a single run_id is provided.
                If the output of multiple runs are to be compated (for example to compare
                the performance of different model on the same task), multiple run_ids are
                passed accordingly.

        Returns:
            An overview of the evaluation. Individual :class:`Evaluation`s will not be
            returned but instead stored in the :class:`EvaluationRepository` provided in the
            __init__.
        """

        def load_overview(run_id: str) -> RunOverview:
            run_overview = self._evaluation_repository.run_overview(run_id)
            if not run_overview:
                raise ValueError(f"No RunOverview found for run-id: {run_id}")
            return run_overview

        if not run_ids:
            raise ValueError("At least one run-id needs to be provided")
        run_overviews = [load_overview(run_id) for run_id in run_ids]
        if not all(
            run_overviews[0].dataset_id == run_overview.dataset_id
            for run_overview in run_overviews
        ):
            raise ValueError(
                f"All run-overviews must reference the same dataset: {run_overviews}"
            )
        eval_id = self._create_evaluation_dataset()
        dataset_id = run_overviews[0].dataset_id
        examples = self._dataset_repository.examples_by_id(
            dataset_id,
            self.input_type(),
            self.expected_output_type(),
        )
        if examples is None:
            raise ValueError(f"Dataset: {dataset_id} not found")
        start = datetime.utcnow()

        examples_zipped: Iterable[tuple[ExampleOutput[Output], ...]] = zip(
            *(
                self._evaluation_repository.example_outputs(
                    run_overview.id, self.output_type()
                )
                for run_overview in run_overviews
            ),
            strict=True,
        )
        for example_outputs in examples_zipped:
            if not any(
                isinstance(output.output, FailedExampleRun)
                for output in example_outputs
            ):
                example_id = example_outputs[0].example_id
                assert all(
                    example_output.example_id == example_id
                    for example_output in example_outputs
                )
                example = self._dataset_repository.example(
                    dataset_id,
                    example_id,
                    self.input_type(),
                    self.expected_output_type(),
                )
                assert example is not None
                self.evaluate(
                    example,
                    eval_id,
                    *[
                        example_output.output
                        for example_output in example_outputs
                        if not isinstance(example_output.output, FailedExampleRun)
                    ],
                )

        partial_overview = PartialEvaluationOverview(
            run_overviews=run_overviews, id=eval_id, start=start
        )
        self._evaluation_repository.store_evaluation_overview(partial_overview)

        return partial_overview

    @final
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

        evaluation_overview = self._evaluation_repository.evaluation_overview(
            eval_id, PartialEvaluationOverview
        )
        if not evaluation_overview:
            raise ValueError(
                f"No PartialEvaluationOverview found for eval-id: {eval_id}"
            )
        example_evaluations = self._evaluation_repository.example_evaluations(
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
        self._evaluation_repository.store_evaluation_overview(run_overview)
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
        evaluation_repository: EvaluationRepository,
        dataset_repository: DatasetRepository,
    ) -> None:
        super().__init__(evaluation_repository, dataset_repository)

    @abstractmethod
    def do_evaluate(
        self,
        input: Input,
        expected_output: ExpectedOutput,
        *output: Output,
    ) -> Evaluation:
        """Executes the evaluation for this use-case.

        Responsible for comparing the input & expected output of a task to the
        actually generated output.

        Args:
            input: The input that was passed to the :class:`Task` to produce the output.
            expected_output: Output that is compared to the generated output.
            output: Output of the :class:`Task` that shall be evaluated.

        Returns:
            The metrics that come from the evaluated :class:`Task`.
        """
        pass

    @final
    def evaluate(
        self, example: Example[Input, ExpectedOutput], eval_id: str, *output: Output
    ) -> None:
        try:
            result: Evaluation | FailedExampleEvaluation = self.do_evaluate(
                example.input,
                example.expected_output,
                *output,
            )
        except Exception as e:
            result = FailedExampleEvaluation.from_exception(e)
        self._evaluation_repository.store_example_evaluation(
            eval_id, ExampleEvaluation(example_id=example.id, result=result)
        )

    @final
    def run_and_evaluate(
        self,
        task: Task[Input, Output],
        input: Input,
        expected_output: ExpectedOutput,
        tracer: Tracer,
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
            output = task.run(input, tracer)
            return self.do_evaluate(input, expected_output, output)
        except Exception as e:
            return FailedExampleEvaluation.from_exception(e)

    @final
    def evaluate_dataset(
        self, *run_ids: str
    ) -> EvaluationOverview[AggregatedEvaluation]:
        """Evaluates an entire dataset in a threaded manner and aggregates the results into an `AggregatedEvaluation`.

        This will call the `run` method for each example in the dataset.
        Finally, it will call the `aggregate` method and return the aggregated results.

        Args:
            dataset_id: id of the dataset that will be used to evaluate a :class:`Task`.
                The actual data is loaded from the :class:`DatasetRepository` passed to `__init__`
            tracer: Optional tracer used for extra tracing.
                Traces are always saved in the evaluation repository.

        Returns:
            The aggregated results of an evaluation run with a dataset.
        """
        partial_evaluation_overview = self.evaluate_runs(*run_ids)
        return self.aggregate_evaluation(partial_evaluation_overview.id)


class ArgillaEvaluationRepository(EvaluationRepository):
    """Evaluation repository used for the :class:`ArgillaEvaluator`.

    Wraps an :class:`Evaluator`.
    Does not support storing evaluations, since the ArgillaEvaluator does not do automated evaluations.

    Args:
        evaluation_repository: repository to wrap.
        argilla_client: client used to connect to Argilla.
    """

    def __init__(
        self, evaluation_repository: EvaluationRepository, argilla_client: ArgillaClient
    ) -> None:
        super().__init__()
        self._evaluation_repository = evaluation_repository
        self._client = argilla_client

    def run_ids(self) -> Sequence[str]:
        return self._evaluation_repository.run_ids()

    def eval_ids(self) -> Sequence[str]:
        return self._evaluation_repository.eval_ids()

    def example_outputs(
        self, run_id: str, output_type: type[Output]
    ) -> Iterable[ExampleOutput[Output]]:
        return self._evaluation_repository.example_outputs(run_id, output_type)

    def store_example_output(
        self, run_id: str, example_output: ExampleOutput[Output]
    ) -> None:
        return self._evaluation_repository.store_example_output(run_id, example_output)

    def example_trace(self, run_id: str, example_id: str) -> Optional[ExampleTrace]:
        return self._evaluation_repository.example_trace(run_id, example_id)

    def example_tracer(self, run_id: str, example_id: str) -> Tracer:
        return self._evaluation_repository.example_tracer(run_id, example_id)

    def example_evaluation(
        self, eval_id: str, example_id: str, evaluation_type: type[Evaluation]
    ) -> Optional[ExampleEvaluation[Evaluation]]:
        return self._evaluation_repository.example_evaluation(
            eval_id, example_id, evaluation_type
        )

    def store_example_evaluation(
        self, _: str, __: ExampleEvaluation[Evaluation]
    ) -> None:
        raise TypeError(
            "ArgillaEvaluationRepository does not support storing evaluations."
        )

    def example_evaluations(
        self, eval_id: str, eval_type: type[Evaluation]
    ) -> Sequence[ExampleEvaluation[Evaluation]]:
        assert eval_type == ArgillaEvaluation
        # Mypy does not derive that the return type is always ExampleEvaluation with ArgillaEvaluation
        return [ExampleEvaluation(example_id=e.example_id, result=e) for e in self._client.evaluations(eval_id)]  # type: ignore

    def failed_example_evaluations(
        self, eval_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleEvaluation[Evaluation]]:
        return self._evaluation_repository.failed_example_evaluations(
            eval_id, evaluation_type
        )

    def evaluation_overview(
        self, eval_id: str, overview_type: type[EvaluationOverviewType]
    ) -> EvaluationOverviewType | None:
        return self._evaluation_repository.evaluation_overview(eval_id, overview_type)

    def store_evaluation_overview(self, overview: PartialEvaluationOverview) -> None:
        return self._evaluation_repository.store_evaluation_overview(overview)

    def run_overview(self, run_id: str) -> RunOverview | None:
        return self._evaluation_repository.run_overview(run_id)

    def store_run_overview(self, overview: RunOverview) -> None:
        return self._evaluation_repository.store_run_overview(overview)


class ArgillaEvaluator(
    BaseEvaluator[
        Input, Output, ExpectedOutput, ArgillaEvaluation, AggregatedEvaluation
    ],
    ABC,
):
    """Evaluator used to integrate with Argilla (https://github.com/argilla-io/argilla).

    Use this evaluator if you would like to easily do human eval.
    This evaluator runs a dataset and sends the input and output to Argilla to be evaluated.
    After they have been evaluated, you can fetch the results by using the `aggregate_evaluation` method.

    Args:
        task: The task that will be evaluated.
        repository: The repository that will be used to store evaluation results.
        workspace_id: The workspace id to save the datasets in. Has to be created before in Argilla.
        fields: The Argilla fields of the dataset.
        questions: The questions that will be presented to the human evaluators.
    """

    def __init__(
        self,
        evaluation_repository: ArgillaEvaluationRepository,
        dataset_repository: DatasetRepository,
        workspace_id: str,
        fields: Sequence[Field],
        questions: Sequence[Question],
    ) -> None:
        super().__init__(evaluation_repository, dataset_repository)
        self._workspace_id = workspace_id
        self._fields = fields
        self._questions = questions
        self._client = evaluation_repository._client

    def evaluation_type(self) -> type[ArgillaEvaluation]:
        return ArgillaEvaluation

    @final
    def _create_evaluation_dataset(self) -> str:
        return self._client.create_dataset(
            self._workspace_id,
            str(uuid4()),
            self._fields,
            self._questions,
        )

    @final
    def partial_evaluate_dataset(self, *run_ids: str) -> PartialEvaluationOverview:
        """Evaluates an entire :class:`Dataset` in a threaded manner and pushes the results to Argilla.

        Args:
            dataset: Dataset that will be used to evaluate a :class:`Task`.
            tracer: Optional tracer used for extra tracing.
                Traces are always saved in the evaluation repository.

        Returns:
            An overview of how the run went (e.g. how many examples failed).
        """
        return self.evaluate_runs(*run_ids)

    @abstractmethod
    def _to_record(
        self, example: Example[Input, ExpectedOutput], *output: Output
    ) -> Sequence[RecordData]:
        """This method is responsible for translating the `Example` and `Output` of the task to :class:`RecordData`

        Args:
            example: The example to be translated.
            output: The output of the example that was run.
        """
        ...

    @final
    def evaluate(
        self, example: Example[Input, ExpectedOutput], eval_id: str, *output: Output
    ) -> None:
        records = self._to_record(example, *output)
        for record in records:
            self._client.add_record(eval_id, record)
