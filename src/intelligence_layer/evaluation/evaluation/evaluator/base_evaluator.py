from abc import ABC
from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from typing import (
    Generic,
    Optional,
    TypeVar,
    cast,
    get_args,
    get_origin,
)

from intelligence_layer.core import Input, Output
from intelligence_layer.evaluation.dataset.dataset_repository import DatasetRepository
from intelligence_layer.evaluation.dataset.domain import Example, ExpectedOutput
from intelligence_layer.evaluation.evaluation.domain import Evaluation
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


class EvaluationLogicBase(Generic[Input, Output, ExpectedOutput, Evaluation]):
    pass


class EvaluatorBase(Generic[Input, Output, ExpectedOutput, Evaluation], ABC):
    """Base class for Evaluators that can handle automatic evaluation scenarios.

    Provides methods for type inference and loading data from the repositories.

    Arguments:
        dataset_repository: The repository with the examples that will be taken for the evaluation.
        run_repository: The repository of the runs to evaluate.
        evaluation_repository: The repository that will be used to store evaluation results.
        description: Human-readable description for the evaluator.
        evaluation_logic: The logic to use for evaluation.

    Generics:
        Input: Type to be passed to the :class:`Task` as input. Part of `Example`.
        Output: Type of the output of the :class:`Task` to be evaluated. Part of `ExampleOutput`
        ExpectedOutput: Type that the `Output` will be compared against. Part of `Example`.
        Evaluation: Type of the metrics that come from the evaluated :class:`Task`. Part of `ExampleEvaluation`
    """

    def __init__(
        self,
        dataset_repository: DatasetRepository,
        run_repository: RunRepository,
        evaluation_repository: EvaluationRepository,
        description: str,
        evaluation_logic: EvaluationLogicBase[
            Input, Output, ExpectedOutput, Evaluation
        ],
    ) -> None:
        self._dataset_repository = dataset_repository
        self._run_repository = run_repository
        self._evaluation_repository = evaluation_repository
        self._evaluation_logic = evaluation_logic

        self.description = description

    @cached_property
    def _get_types(self) -> Mapping[str, type]:
        """Type magic function that gets the actual types of the generic parameters.

        Traverses the inheritance history of `EvaluationLogicBase`-subclasses to find an actual type every time a TypeVar is replaced.

        Returns:
            Name of generic parameter to the type found.
        """

        def is_eligible_subclass(parent: type) -> bool:
            return hasattr(parent, "__orig_bases__") and issubclass(
                parent, EvaluationLogicBase
            )

        def update_types() -> None:
            num_types_set = 0
            for current_index, current_type in enumerate(current_types):
                if type(current_type) is not TypeVar:
                    type_var_count = num_types_set - 1
                    final_element_index = -1
                    for element_index, element in enumerate(type_list):
                        final_element_index = element_index
                        if type(element) is TypeVar:
                            type_var_count += 1
                        if type_var_count == current_index:
                            break
                    assert type_var_count == current_index
                    type_list[final_element_index] = current_type
                    num_types_set += 1

        # mypy does not know __orig_bases__
        base_types = EvaluationLogicBase.__orig_bases__[0]  # type: ignore
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
                if origin is None or not issubclass(origin, EvaluationLogicBase):
                    continue
                current_types = list(get_args(base))
                update_types()

        return {
            name: param_type
            for name, param_type in zip(
                (a.__name__ for a in get_args(base_types)), type_list, strict=True
            )
            if type(param_type) is not TypeVar
        }

    def input_type(self) -> type[Input]:
        """Returns the type of the evaluated task's input.

        This can be used to retrieve properly typed :class:`Example`s of a dataset
        from a :class:`DatasetRepository`.

        Returns:
            The type of the evaluated task's input.
        """
        try:
            input_type = self._get_types["Input"]
        except KeyError:
            raise TypeError(
                f"Alternatively overwrite input_type() in {type(self)}"
            ) from None
        return cast(type[Input], input_type)

    def output_type(self) -> type[Output]:
        """Returns the type of the evaluated task's output.

        This can be used to retrieve properly typed outputs of an evaluation run
        from a :class:`RunRepository`.

        Returns:
            The type of the evaluated task's output.
        """
        try:
            output_type = self._get_types["Output"]
        except KeyError:
            raise TypeError(
                f"Alternatively overwrite output_type() in {type(self)}"
            ) from None
        return cast(type[Output], output_type)

    def expected_output_type(self) -> type[ExpectedOutput]:
        """Returns the type of the evaluated task's expected output.

        This can be used to retrieve properly typed :class:`Example`s of a dataset
        from a :class:`DatasetRepository`.

        Returns:
            The type of the evaluated task's expected output.
        """
        try:
            expected_output_type = self._get_types["ExpectedOutput"]
        except KeyError:
            raise TypeError(
                f"Alternatively overwrite expected_output_type() in {type(self)}"
            ) from None
        return cast(type[ExpectedOutput], expected_output_type)

    def evaluation_type(self) -> type[Evaluation]:
        """Returns the type of the evaluation result of an example.

        This can be used to retrieve properly typed evaluations of an evaluation run
        from an :class:`EvaluationRepository`

        Returns:
            Returns the type of the evaluation result of an example.
        """
        try:
            evaluation_type = self._get_types["Evaluation"]
        except KeyError:
            raise TypeError(
                f"Alternatively overwrite evaluation_type() in {type(self)}"
            ) from None
        return cast(type[Evaluation], evaluation_type)

    def _load_run_overviews(self, *run_ids: str) -> set[RunOverview]:
        if not run_ids:
            raise ValueError("At least one run-id needs to be provided")
        run_overviews = set()
        for run_id in run_ids:
            run_overview = self._run_repository.run_overview(run_id)
            if not run_overview:
                raise ValueError(f"No RunOverview found for run-id: {run_id}")

            run_overviews.add(run_overview)
        return run_overviews

    def _raise_if_overviews_have_different_dataset(
        self, run_overviews: set[RunOverview]
    ) -> None:
        if any(
            next(iter(run_overviews)).dataset_id != run_overview.dataset_id
            for run_overview in run_overviews
        ):
            raise ValueError(
                f"All run overviews must reference the same dataset: {run_overviews}"
            )

    def _raise_if_overviews_have_different_number_of_runs(
        self, run_overviews: set[RunOverview]
    ) -> None:
        run_overview_list = list(run_overviews)
        if any(
            run_overview_list[0].failed_example_count
            + run_overview_list[0].successful_example_count
            != run_overview.successful_example_count + run_overview.failed_example_count
            for run_overview in run_overview_list
        ):
            raise ValueError(
                f"All run overviews must contain the same number of items: {run_overviews}"
            )

    def _retrieve_example_outputs(
        self, run_overviews: set[RunOverview]
    ) -> Iterable[tuple[ExampleOutput[Output], ...]]:
        # this uses the assumption that the example outputs are sorted and there is never a missing example
        example_outputs_for_example: Iterable[tuple[ExampleOutput[Output], ...]] = zip(
            *(
                self._run_repository.example_outputs(
                    run_overview.id, self.output_type()
                )
                for run_overview in run_overviews
            ),
            strict=True,
        )

        return example_outputs_for_example

    def _retrieve_examples(
        self, dataset_id: str
    ) -> Iterable[Example[Input, ExpectedOutput]]:
        examples = self._dataset_repository.examples(
            dataset_id,
            self.input_type(),
            self.expected_output_type(),
        )
        if examples is None:
            raise ValueError(f"Dataset: {dataset_id} not found")

        return examples

    def _generate_evaluation_inputs(
        self,
        examples: Iterable[Example[Input, ExpectedOutput]],
        example_outputs_for_example: Iterable[tuple[ExampleOutput[Output], ...]],
        skip_example_on_any_failure: bool,
        num_examples: Optional[int],
    ) -> Iterable[
        tuple[
            Example[Input, ExpectedOutput],
            Sequence[SuccessfulExampleOutput[Output]],
        ]
    ]:
        current_example = 0

        for example, example_outputs in zip(
            examples, example_outputs_for_example, strict=False
        ):
            if any(
                example.id != example_output.example_id
                for example_output in example_outputs
            ):
                raise ValueError(
                    "The ids of example and output do not match. Therefore, the evaluation cannot continue.\n"
                    + f"example id: {example.id}, output id: {example_outputs}."
                )
            if skip_example_on_any_failure and any(
                isinstance(output.output, FailedExampleRun)
                for output in example_outputs
            ):
                continue

            successful_example_outputs = [
                cast(SuccessfulExampleOutput[Output], output)
                for output in example_outputs
                if not isinstance(output.output, FailedExampleRun)
            ]

            if num_examples and current_example >= num_examples:
                break
            current_example += 1

            yield (
                example,
                successful_example_outputs,
            )

    def _retrieve_eval_logic_input(
        self,
        run_overviews: set[RunOverview],
        skip_example_on_any_failure: bool,
        num_examples: Optional[int] = None,
    ) -> Iterable[
        tuple[
            Example[Input, ExpectedOutput],
            Sequence[SuccessfulExampleOutput[Output]],
        ]
    ]:
        """Create pairings of :class:`Example` and all corresponding :class:`ExampleOutputs`.

        In case an Example is matched with a FailedExampleRun, that example is skipped, even if
        there are other successful ExampleOutputs present for this example.

        Args:
            run_overviews: Run overviews to gather data from.
            skip_example_on_any_failure: Skip example on any failure.
            num_examples: Maximum amount of examples to gather. Defaults to None.

        Returns:
            Iterable over pairs of :class:`Example` and all corresponding :class:`ExampleOutputs`.
        """
        self._raise_if_overviews_have_different_dataset(run_overviews)
        self._raise_if_overviews_have_different_number_of_runs(run_overviews)
        example_outputs_for_example = self._retrieve_example_outputs(run_overviews)
        dataset_id = next(iter(run_overviews)).dataset_id
        examples = self._retrieve_examples(dataset_id)
        return self._generate_evaluation_inputs(
            examples,
            example_outputs_for_example,
            skip_example_on_any_failure,
            num_examples,
        )

    def failed_evaluations(
        self, evaluation_id: str
    ) -> Iterable[EvaluationLineage[Input, ExpectedOutput, Output, Evaluation]]:
        """Returns the `EvaluationLineage` objects for all failed example evaluations that belong to the given evaluation ID.

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
