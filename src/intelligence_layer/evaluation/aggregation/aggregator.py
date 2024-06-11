from abc import ABC, abstractmethod
from functools import cached_property
from typing import (
    Callable,
    Generic,
    Iterable,
    Iterator,
    Mapping,
    TypeVar,
    cast,
    final,
    get_args,
    get_origin,
)
from uuid import uuid4

from intelligence_layer.core import utc_now
from intelligence_layer.evaluation.aggregation.aggregation_repository import (
    AggregationRepository,
)
from intelligence_layer.evaluation.aggregation.domain import (
    AggregatedEvaluation,
    AggregationOverview,
)
from intelligence_layer.evaluation.evaluation.domain import (
    Evaluation,
    EvaluationOverview,
    FailedExampleEvaluation,
)
from intelligence_layer.evaluation.evaluation.evaluation_repository import (
    EvaluationRepository,
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


class AggregationLogic(ABC, Generic[Evaluation, AggregatedEvaluation]):
    @abstractmethod
    def aggregate(self, evaluations: Iterable[Evaluation]) -> AggregatedEvaluation:
        """`Evaluator`-specific method for aggregating individual `Evaluations` into report-like `Aggregated Evaluation`.

        This method is responsible for taking the results of an evaluation run and aggregating all the results.
        It should create an `AggregatedEvaluation` class and return it at the end.

        Args:
            evaluations: The results from running `eval_and_aggregate_runs` with a :class:`Task`.

        Returns:
            The aggregated results of an evaluation run with a :class:`Dataset`.
        """
        ...


class Aggregator(Generic[Evaluation, AggregatedEvaluation]):
    """Aggregator that can handle automatic aggregation of evaluation scenarios.

    This aggregator should be used for automatic eval. A user still has to implement
    :class: `AggregationLogic`.


    Arguments:
        evaluation_repository: The repository that will be used to store evaluation results.
        aggregation_repository: The repository that will be used to store aggregation results.
        description: Human-readable description for the evaluator.
        aggregation_logic: The logic to aggregate the evaluations.

    Generics:
        Evaluation: Interface of the metrics that come from the evaluated :class:`Task`.
        AggregatedEvaluation: The aggregated results of an evaluation run with a :class:`Dataset`.
    """

    def __init__(
        self,
        evaluation_repository: EvaluationRepository,
        aggregation_repository: AggregationRepository,
        description: str,
        aggregation_logic: AggregationLogic[Evaluation, AggregatedEvaluation],
    ) -> None:
        self._evaluation_repository = evaluation_repository
        self._aggregation_repository = aggregation_repository
        self._aggregation_logic = aggregation_logic
        self.description = description

    @cached_property
    def _get_types(self) -> Mapping[str, type]:
        """Type magic function that gets the actual types of the generic parameters.

        Traverses the inheritance history of `AggregationLogic`-subclass to find an actual type every time a TypeVar is replaced.

        Returns:
            Name of generic parameter to the type found.
        """

        def is_eligible_subclass(parent: type) -> bool:
            return hasattr(parent, "__orig_bases__") and issubclass(
                parent, AggregationLogic
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
        base_types = AggregationLogic.__orig_bases__[1]  # type: ignore
        type_list: list[type | TypeVar] = list(get_args(base_types))

        possible_parent_classes = [
            p
            for p in reversed(type(self._aggregation_logic).__mro__)
            if is_eligible_subclass(p)
        ]
        for parent in possible_parent_classes:
            # mypy does not know __orig_bases__
            for base in parent.__orig_bases__:  # type: ignore
                origin = get_origin(base)
                if origin is None or not issubclass(origin, AggregationLogic):
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

    def evaluation_type(self) -> type[Evaluation]:
        """Returns the type of the evaluation result of an example.

        This can be used to retrieve properly typed evaluations of an evaluation run
        from a :class:`EvaluationRepository`

        Returns:
            Returns the type of the evaluation result of an example.
        """
        try:
            evaluation_type = self._get_types["Evaluation"]
        except KeyError:
            raise TypeError(
                f"Alternatively overwrite evaluation_type() in {type(self)}"
            )
        return cast(type[Evaluation], evaluation_type)

    @final
    def aggregate_evaluation(
        self, *eval_ids: str
    ) -> AggregationOverview[AggregatedEvaluation]:
        """Aggregates all evaluations into an overview that includes high-level statistics.

        Aggregates :class:`Evaluation`s according to the implementation of :func:`AggregationLogic.aggregate`.

        Args:
            eval_ids: An overview of the evaluation to be aggregated. Does not include
                actual evaluations as these will be retrieved from the repository.

        Returns:
            An overview of the aggregated evaluation.
        """

        def load_eval_overview(evaluation_id: str) -> EvaluationOverview:
            evaluation_overview = self._evaluation_repository.evaluation_overview(
                evaluation_id
            )
            if not evaluation_overview:
                raise ValueError(
                    f"No PartialEvaluationOverview found for eval-id: {evaluation_id}"
                )
            return evaluation_overview

        evaluation_overviews = frozenset(
            load_eval_overview(evaluation_id) for evaluation_id in set(eval_ids)
        )

        nested_evaluations = [
            self._evaluation_repository.example_evaluations(
                overview.id, self.evaluation_type()
            )
            for overview in evaluation_overviews
        ]
        example_evaluations = [
            evaluation for sublist in nested_evaluations for evaluation in sublist
        ]

        successful_evaluations = CountingFilterIterable(
            (example_eval.result for example_eval in example_evaluations),
            lambda evaluation: not isinstance(evaluation, FailedExampleEvaluation),
        )
        start = utc_now()
        statistics = self._aggregation_logic.aggregate(
            cast(Iterable[Evaluation], successful_evaluations)
        )

        aggregation_overview = AggregationOverview(
            evaluation_overviews=frozenset(evaluation_overviews),
            id=str(uuid4()),
            start=start,
            end=utc_now(),
            successful_evaluation_count=successful_evaluations.included_count(),
            crashed_during_evaluation_count=successful_evaluations.excluded_count(),
            description=self.description,
            statistics=statistics,
        )
        self._aggregation_repository.store_aggregation_overview(aggregation_overview)
        return aggregation_overview
