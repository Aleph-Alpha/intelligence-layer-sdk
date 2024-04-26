import typing
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import (
    Generic,
    Iterable,
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

from tqdm import tqdm

from intelligence_layer.core import Input, Output, utc_now
from intelligence_layer.evaluation.dataset.dataset_repository import DatasetRepository
from intelligence_layer.evaluation.dataset.domain import Example, ExpectedOutput
from intelligence_layer.evaluation.evaluation.domain import (
    Evaluation,
    EvaluationOverview,
    ExampleEvaluation,
    FailedExampleEvaluation,
)
from intelligence_layer.evaluation.evaluation.evaluation_repository import (
    EvaluationRepository,
)
from intelligence_layer.evaluation.infrastructure.repository_navigator import (
    EvaluationLineage,
    RepositoryNavigator,
)
from intelligence_layer.evaluation.run.domain import (
    ExampleOutput,
    FailedExampleRun,
    RunOverview,
    SuccessfulExampleOutput,
)
from intelligence_layer.evaluation.run.run_repository import RunRepository


class EvaluationLogic(ABC, Generic[Input, Output, ExpectedOutput, Evaluation]):
    @abstractmethod
    def do_evaluate(
        self,
        example: Example[Input, ExpectedOutput],
        *output: SuccessfulExampleOutput[Output],
    ) -> Evaluation:
        """Executes the evaluation for this specific example.

        Responsible for comparing the input & expected output of a task to the
        actually generated output.

        Args:
            example: Input data of :class:`Task` to produce the output.
            output: Output of the :class:`Task`.

        Returns:
            The metrics that come from the evaluated :class:`Task`.
        """
        pass


class IncrementalEvaluationLogic(
    EvaluationLogic[Input, Output, ExpectedOutput, Evaluation]
):
    def __init__(self) -> None:
        super().__init__()
        self._previous_run_output_ids: list[set[str]] = []

    def set_previous_run_output_ids(
        self, previous_run_output_ids: list[set[str]]
    ) -> None:
        self._previous_run_output_ids = previous_run_output_ids

    def do_evaluate(
        self,
        example: Example[Input, ExpectedOutput],
        *outputs: SuccessfulExampleOutput[Output],
    ) -> Evaluation:
        """Executes the evaluation for this specific example.

        Responsible for comparing the input & expected output of a task to the
        actually generated output. The difference to the standard :class:`EvaluationLogic`'s `do_evaluate` is that
        this method will separate already processed evaluation from new ones before handing them over to
        `do_incremental_evaluate`.

        Args:
            example: Input data of :class:`Task` to produce the output.
            outputs: Outputs of the :class:`Task`.

        Returns:
            :class:`Evaluation`: The metrics that come from the evaluated :class:`Task`.
        """
        flattened_run_output_ids: set[str] = set()
        evaluated_outputs = []
        for run_output_ids in self._previous_run_output_ids:
            flattened_run_output_ids = flattened_run_output_ids.union(run_output_ids)
            evaluated_outputs.append(
                [output for output in outputs if output.run_id in run_output_ids]
            )

        new_outputs = [
            output
            for output in outputs
            if output.run_id not in flattened_run_output_ids
        ]
        return self.do_incremental_evaluate(example, new_outputs, evaluated_outputs)

    @abstractmethod
    def do_incremental_evaluate(
        self,
        example: Example[Input, ExpectedOutput],
        outputs: list[SuccessfulExampleOutput[Output]],
        already_evaluated_outputs: list[list[SuccessfulExampleOutput[Output]]],
    ) -> Evaluation:
        pass


class SingleOutputEvaluationLogic(
    EvaluationLogic[Input, Output, ExpectedOutput, Evaluation]
):
    @final
    def do_evaluate(
        self,
        example: Example[Input, ExpectedOutput],
        *output: SuccessfulExampleOutput[Output],
    ) -> Evaluation:
        assert len(output) == 1
        return self.do_evaluate_single_output(example, output[0].output)

    @abstractmethod
    def do_evaluate_single_output(
        self, example: Example[Input, ExpectedOutput], output: Output
    ) -> Evaluation:
        pass


class Evaluator(Generic[Input, Output, ExpectedOutput, Evaluation]):
    """Evaluator that can handle automatic evaluation scenarios.

    This evaluator should be used for automatic eval. A user still has to implement
    :class:`EvaluationLogic`.


    Arguments:
        dataset_repository: The repository with the examples that will be taken for the evaluation.
        run_repository: The repository of the runs to evaluate.
        evaluation_repository: The repository that will be used to store evaluation results.
        description: Human-readable description for the evaluator.
        evaluation_logic: The logic to use for evaluation.

    Generics:
        Input: Interface to be passed to the :class:`Task` that shall be evaluated.
        Output: Type of the output of the :class:`Task` to be evaluated.
        ExpectedOutput: Output that is expected from the run with the supplied input.
        Evaluation: Interface of the metrics that come from the evaluated :class:`Task`.
    """

    def __init__(
        self,
        dataset_repository: DatasetRepository,
        run_repository: RunRepository,
        evaluation_repository: EvaluationRepository,
        description: str,
        evaluation_logic: EvaluationLogic[Input, Output, ExpectedOutput, Evaluation],
    ) -> None:
        self._dataset_repository = dataset_repository
        self._run_repository = run_repository
        self._evaluation_repository = evaluation_repository
        self._evaluation_logic = evaluation_logic
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
        base_types = EvaluationLogic.__orig_bases__[1]  # type: ignore
        type_list: list[type | TypeVar] = list(get_args(base_types))
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
                (a.__name__ for a in get_args(base_types)), type_list
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

    def evaluate_runs(
        self,
        *run_ids: str,
        num_examples: Optional[int] = None,
        abort_on_error: bool = False,
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
            abort_on_error: Flag to abort all evaluations when an error occurs. Defaults to False.

        Returns:
            EvaluationOverview: An overview of the evaluation. Individual :class:`Evaluation`s will not be
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
        eval_id = self._evaluation_repository.initialize_evaluation()
        dataset_id = next(iter(run_overviews)).dataset_id
        examples = self._dataset_repository.examples(
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
                successful_example_outputs = [
                    typing.cast(SuccessfulExampleOutput[Output], output)
                    for output in example_outputs
                    if not isinstance(output.output, FailedExampleRun)
                ]
                if not successful_example_outputs:
                    continue
                example_id = successful_example_outputs[0].example_id
                assert all(
                    example_output.example_id == example_id
                    for example_output in successful_example_outputs
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
                    successful_example_outputs,
                )

        with ThreadPoolExecutor(max_workers=10) as executor:
            list(  # the list is needed to consume the iterator returned from the executor.map
                tqdm(
                    executor.map(
                        lambda args: self.evaluate(
                            args[0], args[1], abort_on_error, *args[2]
                        ),
                        generate_evaluation_inputs(),
                    ),
                    desc="Evaluating",
                )
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
    def evaluate(
        self,
        example: Example[Input, ExpectedOutput],
        evaluation_id: str,
        abort_on_error: bool,
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
            if abort_on_error:
                raise e
            print(
                f'FAILED EVALUATION: example "{example.id}", {type(e).__qualname__}: "{e}"'
            )
            result = FailedExampleEvaluation.from_exception(e)
        self._evaluation_repository.store_example_evaluation(
            ExampleEvaluation(
                evaluation_id=evaluation_id, example_id=example.id, result=result
            )
        )

    def failed_evaluations(
        self, evaluation_id: str
    ) -> Iterable[EvaluationLineage[Input, ExpectedOutput, Output, Evaluation]]:
        """Returns the `EvaluationLineage` objects for all failed example evalations that belong to the given evaluation ID.

        Args:
            evaluation_id: The ID of the evaluation overview

        Returns:
            :class:`Iterable` of :class:`EvaluationLineage`s.
        """
        failed_example_evaluations = (
            self._evaluation_repository.failed_example_evaluations(
                evaluation_id, evaluation_type=self.evaluation_type()
            )
        )
        lineages = (
            self.evaluation_lineage(evaluation_id, output.example_id)
            for output in failed_example_evaluations
        )
        return (lineage for lineage in lineages if lineage is not None)

    def evaluation_lineages(
        self, evaluation_id: str
    ) -> Iterable[EvaluationLineage[Input, ExpectedOutput, Output, Evaluation]]:
        """Wrapper for `RepositoryNagivator.evaluation_lineages`.

        Args:
            evaluation_id: The id of the evaluation

        Returns:
            An iterator over all :class:`EvaluationLineage`s for the given evaluation id.
        """
        navigator = RepositoryNavigator(
            self._dataset_repository, self._run_repository, self._evaluation_repository
        )
        return navigator.evaluation_lineages(
            evaluation_id=evaluation_id,
            input_type=self.input_type(),
            expected_output_type=self.expected_output_type(),
            output_type=self.output_type(),
            evaluation_type=self.evaluation_type(),
        )

    def evaluation_lineage(
        self, evaluation_id: str, example_id: str
    ) -> EvaluationLineage[Input, ExpectedOutput, Output, Evaluation] | None:
        """Wrapper for `RepositoryNagivator.evaluation_lineage`.

        Args:
            evaluation_id: The id of the evaluation
            example_id: The id of the example of interest

        Returns:
            The :class:`EvaluationLineage` for the given evaluation id and example id.
            Returns `None` if the lineage is not complete because either an example, a run, or an evaluation does not exist.
        """
        navigator = RepositoryNavigator(
            self._dataset_repository, self._run_repository, self._evaluation_repository
        )
        return navigator.evaluation_lineage(
            evaluation_id=evaluation_id,
            example_id=example_id,
            input_type=self.input_type(),
            expected_output_type=self.expected_output_type(),
            output_type=self.output_type(),
            evaluation_type=self.evaluation_type(),
        )


class IncrementalEvaluator(Evaluator[Input, Output, ExpectedOutput, Evaluation]):
    """:class:`Evaluator` for evaluating additional runs on top of previous evaluations. Intended for use with :class:`IncrementalEvaluationLogic`.

    Args:
        dataset_repository: The repository with the examples that will be taken for the evaluation.
        run_repository: The repository of the runs to evaluate.
        evaluation_repository: The repository that will be used to store evaluation results.
        description: Human-readable description for the evaluator.
        incremental_evaluation_logic: The logic to use for evaluation.

    Generics:
        Input: Interface to be passed to the :class:`Task` that shall be evaluated.
        Output: Type of the output of the :class:`Task` to be evaluated.
        ExpectedOutput: Output that is expected from the run with the supplied input.
        Evaluation: Interface of the metrics that come from the evaluated :class:`Task`.
    """

    def __init__(
        self,
        dataset_repository: DatasetRepository,
        run_repository: RunRepository,
        evaluation_repository: EvaluationRepository,
        description: str,
        incremental_evaluation_logic: IncrementalEvaluationLogic[
            Input, Output, ExpectedOutput, Evaluation
        ],
    ) -> None:
        super().__init__(
            dataset_repository=dataset_repository,
            run_repository=run_repository,
            evaluation_repository=evaluation_repository,
            description=description,
            evaluation_logic=incremental_evaluation_logic,
        )

    def evaluate_additional_runs(
        self,
        *run_ids: str,
        previous_evaluation_ids: Optional[list[str]] = None,
        num_examples: Optional[int] = None,
        abort_on_error: bool = False,
    ) -> EvaluationOverview:
        """Evaluate all runs while considering which runs have already been evaluated according to `previous_evaluation_id`.

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
            previous_evaluation_ids: IDs of previous evaluation to consider
            num_examples: The number of examples which should be evaluated from the given runs.
                Always the first n runs stored in the evaluation repository
            abort_on_error: Flag to abort all evaluations when an error occurs. Defaults to False.

        Returns:
            EvaluationOverview: An overview of the evaluation. Individual :class:`Evaluation`s will not be
            returned but instead stored in the :class:`EvaluationRepository` provided in the
            __init__.
        """

        previous_run_ids = []
        previous_evaluation_ids = previous_evaluation_ids or []

        for previous_evaluation_id in previous_evaluation_ids:
            prev_run_ids: set[str] = set()
            lineages = self.evaluation_lineages(previous_evaluation_id)
            for lineage in lineages:
                for output in lineage.outputs:
                    prev_run_ids.add(output.run_id)
            previous_run_ids.append(prev_run_ids)

        cast(
            IncrementalEvaluationLogic[Input, Output, ExpectedOutput, Evaluation],
            self._evaluation_logic,
        ).set_previous_run_output_ids(previous_run_ids)
        return super().evaluate_runs(
            *run_ids, num_examples=num_examples, abort_on_error=abort_on_error
        )

    def evaluate_runs(
        self,
        *run_ids: str,
        num_examples: Optional[int] = None,
        abort_on_error: bool = False,
    ) -> EvaluationOverview:
        cast(
            IncrementalEvaluationLogic[Input, Output, ExpectedOutput, Evaluation],
            self._evaluation_logic,
        ).set_previous_run_output_ids([])
        return super().evaluate_runs(
            *run_ids, num_examples=num_examples, abort_on_error=abort_on_error
        )
