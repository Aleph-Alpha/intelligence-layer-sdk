from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Optional

from intelligence_layer.evaluation.aggregation.domain import (
    AggregatedEvaluation,
    AggregationOverview,
)


class AggregationRepository(ABC):
    """Base aggregation repository interface.

    Provides methods to store and load aggregated evaluation results: :class:`AggregationOverview`.
    """

    @abstractmethod
    def store_aggregation_overview(
        self, aggregation_overview: AggregationOverview[AggregatedEvaluation]
    ) -> None:
        """Stores an :class:`AggregationOverview`.

        Args:
            aggregation_overview: The aggregated results to be persisted.
        """
        ...

    @abstractmethod
    def aggregation_overview(
        self, aggregation_id: str, aggregation_type: type[AggregatedEvaluation]
    ) -> Optional[AggregationOverview[AggregatedEvaluation]]:
        """Returns an :class:`AggregationOverview` for the given ID.

        Args:
            aggregation_id: ID of the aggregation overview to retrieve.
            aggregation_type: Type of the aggregation.

        Returns:
            :class:`EvaluationOverview` if it was found, `None` otherwise.
        """
        ...

    def aggregation_overviews(
        self, aggregation_type: type[AggregatedEvaluation]
    ) -> Iterable[AggregationOverview[AggregatedEvaluation]]:
        """Returns all :class:`AggregationOverview`s sorted by their ID.

        Args:
            aggregation_type: Type of the aggregation.

        Yields:
            :class:`AggregationOverview`s.
        """
        for aggregation_id in self.aggregation_overview_ids():
            aggregation_overview = self.aggregation_overview(
                aggregation_id, aggregation_type
            )
            if aggregation_overview is not None:
                yield aggregation_overview

    @abstractmethod
    def aggregation_overview_ids(self) -> Sequence[str]:
        """Returns sorted IDs of all stored :class:`AggregationOverview`s.

        Returns:
            A :class:`Sequence` of the :class:`AggregationOverview` IDs.
        """
        pass
