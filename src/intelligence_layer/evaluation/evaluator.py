from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import (
    Callable,
    Generic,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    cast,
    final,
    get_args,
    get_origin,
)
from uuid import uuid4

from tqdm import tqdm

from intelligence_layer.core.task import Input, Output
from intelligence_layer.core.tracer import utc_now
from intelligence_layer.evaluation.base_logic import AggregationLogic, EvaluationLogic
from intelligence_layer.evaluation.data_storage.aggregation_repository import (
    AggregationRepository,
)
from intelligence_layer.evaluation.data_storage.dataset_repository import (
    DatasetRepository,
)
from intelligence_layer.evaluation.data_storage.evaluation_repository import (
    EvaluationRepository,
)
from intelligence_layer.evaluation.data_storage.run_repository import RunRepository
from intelligence_layer.evaluation.domain import (
    AggregatedEvaluation,
    AggregationOverview,
    Evaluation,
    EvaluationOverview,
    Example,
    ExampleEvaluation,
    ExampleOutput,
    ExpectedOutput,
    FailedExampleEvaluation,
    FailedExampleRun,
    RunOverview,
    SuccessfulExampleOutput,
)

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
    Generic[Input, Output, ExpectedOutput, Evaluation, AggregatedEvaluation]
):
    """Evaluator that can handle automatic evaluation scenarios.

    This evaluator should be used for automatic eval. A user still has to implement
    :class:`EvaluationLogic` and :class: `AggregationLogic`.


    Arguments:
        dataset_repository: The repository with the examples that will be taken for the evaluation.
        run_repository: The repository of the runs to evaluate.
        evaluation_repository: The repository that will be used to store evaluation results.
        aggregation_repository: The repository that will be used to store aggregation results.
        description: Human-readable description for the evaluator.
        evaluation_logic: The logic to use for evaluation.
        aggregation_logic: The logic to aggregate the evaluations.

    Generics:
        Input: Interface to be passed to the :class:`Task` that shall be evaluated.
        Output: Type of the output of the :class:`Task` to be evaluated.
        ExpectedOutput: Output that is expected from the run with the supplied input.
        Evaluation: Interface of the metrics that come from the evaluated :class:`Task`.
        AggregatedEvaluation: The aggregated results of an evaluation run with a :class:`Dataset`.
    """

    def __init__(
        self,
        dataset_repository: DatasetRepository,
        run_repository: RunRepository,
        evaluation_repository: EvaluationRepository,
        aggregation_repository: AggregationRepository,
        description: str,
        evaluation_logic: EvaluationLogic[Input, Output, ExpectedOutput, Evaluation],
        aggregation_logic: AggregationLogic[Evaluation, AggregatedEvaluation],
    ) -> None:
        self._dataset_repository = dataset_repository
        self._run_repository = run_repository
        self._evaluation_repository = evaluation_repository
        self._aggregation_repository = aggregation_repository

        self._evaluation_logic = evaluation_logic
        self._aggregation_logic = aggregation_logic
        self.description = description

    @lru_cache(maxsize=1)
    def _get_types(self) -> Mapping[str, type]:
        """Type magic function that gets the actual types of the generic parameters.

        Traverses the inheritance history of `BaseEvaluator`-subclass to find an actual type every time a TypeVar is replaced.

        Returns:
            Name of generic parameter to the type found.
        """

        def is_eligible_subclass(parent: type) -> bool:
            return hasattr(parent, "__orig_bases__") and issubclass(
                parent, EvaluationLogic
            )

        def update_types() -> None:
            num_types_set = 0
            for current_index, current_type in enumerate(current_types):
                if type(current_type) is not TypeVar:
                    type_var_count = num_types_set - 1
                    for element_index, element in enumerate(type_list):
                        if type(element) is TypeVar:
                            type_var_count += 1
                        if type_var_count == current_index:
                            break
                    assert type_var_count == current_index
                    type_list[element_index] = current_type
                    num_types_set += 1

        # mypy does not know __orig_bases__
        base_evaluator_bases = EvaluationLogic.__orig_bases__[1]  # type: ignore
        type_list: list[type | TypeVar] = list(get_args(base_evaluator_bases))
        possible_parent_classes = [
            p
            for p in reversed(type(self._evaluation_logic).__mro__)
            if is_eligible_subclass(p)
        ]
        for parent in possible_parent_classes:
            # mypy does not know __orig_bases__
            for base in parent.__orig_bases__:  # type: ignore
                origin = get_origin(base)
                if origin is None or not issubclass(origin, EvaluationLogic):
                    continue
                current_types = list(get_args(base))
                update_types()

        return {
            name: param_type
            for name, param_type in zip(
                (a.__name__ for a in get_args(base_evaluator_bases)), type_list
            )
            if type(param_type) is not TypeVar
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

    @final
    def evaluate_runs(
        self, *run_ids: str, num_examples: Optional[int] = None
    ) -> EvaluationOverview:
        """Evaluates all generated outputs in the run.

        For each set of successful outputs in the referenced runs,
        :func:`EvaluationLogic.do_evaluate` is called and eval metrics are produced &
        stored in the provided :class:`EvaluationRepository`.

        Args:
            run_ids: The runs to be evaluated. Each run is expected to have the same
                dataset as input (which implies their tasks have the same input-type)
                and their tasks have the same output-type. For each example in the
                dataset referenced by the runs the outputs of all runs are collected
                and if all of them were successful they are passed on to the implementation
                specific evaluation. The method compares all run of the provided ids to each other.
            num_examples: The number of examples which should be evaluated from the given runs.
                Always the first n runs stored in the evaluation repository

        Returns:
            An overview of the evaluation. Individual :class:`Evaluation`s will not be
            returned but instead stored in the :class:`EvaluationRepository` provided in the
            __init__.
        """

        def load_run_overview(run_id: str) -> RunOverview:
            run_overview = self._run_repository.run_overview(run_id)
            if not run_overview:
                raise ValueError(f"No RunOverview found for run-id: {run_id}")
            return run_overview

        if not run_ids:
            raise ValueError("At least one run-id needs to be provided")
        run_overviews = frozenset(load_run_overview(run_id) for run_id in run_ids)
        if not all(
            next(iter(run_overviews)).dataset_id == run_overview.dataset_id
            for run_overview in run_overviews
        ):
            raise ValueError(
                f"All run-overviews must reference the same dataset: {run_overviews}"
            )
        eval_id = self._evaluation_repository.create_evaluation_dataset()
        dataset_id = next(iter(run_overviews)).dataset_id
        examples = self._dataset_repository.examples_by_id(
            dataset_id,
            self.input_type(),
            self.expected_output_type(),
        )
        if examples is None:
            raise ValueError(f"Dataset: {dataset_id} not found")
        start = utc_now()

        examples_zipped: Iterable[tuple[ExampleOutput[Output], ...]] = zip(
            *(
                self._run_repository.example_outputs(
                    run_overview.id, self.output_type()
                )
                for run_overview in run_overviews
            ),
            strict=True,
        )

        def generate_evaluation_inputs() -> (
            Iterable[
                Tuple[
                    Example[Input, ExpectedOutput],
                    str,
                    Sequence[SuccessfulExampleOutput[Output]],
                ]
            ]
        ):
            current_example = 0
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

                    if num_examples and current_example >= num_examples:
                        break
                    current_example += 1

                    yield (
                        example,
                        eval_id,
                        [
                            SuccessfulExampleOutput(
                                run_id=example_output.run_id,
                                example_id=example_output.example_id,
                                output=example_output.output,
                            )
                            for example_output in example_outputs
                            if not isinstance(example_output.output, FailedExampleRun)
                        ],
                    )

        def evaluate(
            args: Tuple[
                Example[Input, ExpectedOutput],
                str,
                Sequence[SuccessfulExampleOutput[Output]],
            ],
        ) -> None:
            example, eval_id, example_outputs = args
            self.evaluate(example, eval_id, *example_outputs)

        with ThreadPoolExecutor(max_workers=10) as executor:
            tqdm(
                executor.map(evaluate, generate_evaluation_inputs()),
                desc="Evaluating",
            )

        partial_overview = EvaluationOverview(
            run_overviews=run_overviews,
            id=eval_id,
            start=start,
            description=self.description,
        )
        self._evaluation_repository.store_evaluation_overview(partial_overview)

        return partial_overview

    @final
    def aggregate_evaluation(
        self, *eval_ids: str
    ) -> AggregationOverview[AggregatedEvaluation]:
        """Aggregates all evaluations into an overview that includes high-level statistics.

        Aggregates :class:`Evaluation`s according to the implementation of :func:`BaseEvaluator.aggregate`.

        Args:
            evaluation_overview: An overview of the evaluation to be aggregated. Does not include
                actual evaluations as these will be retrieved from the repository.

        Returns:
            An overview of the aggregated evaluation.
        """

        def load_eval_overview(eval_id: str) -> EvaluationOverview:
            evaluation_overview = self._evaluation_repository.evaluation_overview(
                eval_id
            )
            if not evaluation_overview:
                raise ValueError(
                    f"No PartialEvaluationOverview found for eval-id: {eval_id}"
                )
            return evaluation_overview

        evaluation_overviews = frozenset(load_eval_overview(id) for id in set(eval_ids))

        nested_evaluations = [
            self._evaluation_repository.example_evaluations(
                overview.id, self.evaluation_type()
            )
            for overview in evaluation_overviews
        ]
        example_evaluations = [
            eval for sublist in nested_evaluations for eval in sublist
        ]

        successful_evaluations = CountingFilterIterable(
            (example_eval.result for example_eval in example_evaluations),
            lambda evaluation: not isinstance(evaluation, FailedExampleEvaluation),
        )
        id = str(uuid4())
        start = utc_now()
        statistics = self._aggregation_logic.aggregate(
            cast(Iterable[Evaluation], successful_evaluations)
        )

        aggregation_overview = AggregationOverview(
            evaluation_overviews=frozenset(evaluation_overviews),
            id=id,
            start=start,
            end=utc_now(),
            successful_evaluation_count=successful_evaluations.included_count(),
            crashed_during_eval_count=successful_evaluations.excluded_count(),
            description=self.description,
            statistics=statistics,
        )
        self._aggregation_repository.store_aggregation_overview(aggregation_overview)
        return aggregation_overview

    @final
    def evaluate(
        self,
        example: Example[Input, ExpectedOutput],
        eval_id: str,
        *example_outputs: SuccessfulExampleOutput[Output],
    ) -> None:
        try:
            result: Evaluation | FailedExampleEvaluation = (
                self._evaluation_logic.do_evaluate(
                    example,
                    *example_outputs,
                )
            )
        except Exception as e:
            result = FailedExampleEvaluation.from_exception(e)
        self._evaluation_repository.store_example_evaluation(
            ExampleEvaluation(eval_id=eval_id, example_id=example.id, result=result)
        )

    @final
    def eval_and_aggregate_runs(
        self, *run_ids: str
    ) -> AggregationOverview[AggregatedEvaluation]:
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
