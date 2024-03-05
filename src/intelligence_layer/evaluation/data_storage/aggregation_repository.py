from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Sequence

from intelligence_layer.evaluation.data_storage.utils import FileBasedRepository
from intelligence_layer.evaluation.domain import (
    AggregatedEvaluation,
    AggregationOverview,
)


class AggregationRepository(ABC):
    @abstractmethod
    def store_aggregation_overview(
        self, aggregation_overview: AggregationOverview[AggregatedEvaluation]
    ) -> None:
        """Stores an :class:`AggregationOverview` in the repository.

        Args:
            aggregation_overview: The overview to be persisted.
        """
        ...

    @abstractmethod
    def aggregation_overview(
        self, aggregation_id: str, aggregation_type: type[AggregatedEvaluation]
    ) -> Optional[AggregationOverview[AggregatedEvaluation]]:
        """Returns a specific instance of :class:`AggregationOverview` of a given run

        Args:
            aggregation_id: Identifier of the aggregation overview
            aggregation_type: Type of the aggregation

        Returns:
            :class:`EvaluationOverview` if one was found, `None` otherwise.
        """
        ...


class FileAggregationRepository(AggregationRepository, FileBasedRepository):
    def store_aggregation_overview(
        self, aggregation_overview: AggregationOverview[AggregatedEvaluation]
    ) -> None:
        self.write_utf8(
            self._aggregation_overview_path(aggregation_overview.id),
            aggregation_overview.model_dump_json(indent=2),
        )

    def aggregation_overview(
        self, aggregation_id: str, aggregation_type: type[AggregatedEvaluation]
    ) -> Optional[AggregationOverview[AggregatedEvaluation]]:
        file_path = self._aggregation_overview_path(aggregation_id)
        if not file_path.exists():
            return None
        content = self.read_utf8(file_path)
        return AggregationOverview[aggregation_type].model_validate_json(  # type:ignore
            content
        )

    def aggregation_ids(self) -> Sequence[str]:
        return [path.stem for path in self._aggregation_root_directory().glob("*.json")]

    def _aggregation_root_directory(self) -> Path:
        path = self._root_directory / "aggregation"
        path.mkdir(exist_ok=True)
        return path

    def _aggregation_overview_path(self, aggregation_id: str) -> Path:
        return (self._aggregation_root_directory() / aggregation_id).with_suffix(
            ".json"
        )


class InMemoryAggregationRepository(AggregationRepository):
    def __init__(self) -> None:
        super().__init__()
        self._aggregation_overviews: dict[str, AggregationOverview[Any]] = dict()

    def store_aggregation_overview(
        self, aggregation_overview: AggregationOverview[AggregatedEvaluation]
    ) -> None:
        self._aggregation_overviews[aggregation_overview.id] = aggregation_overview

    def aggregation_overview(
        self, aggregation_id: str, aggregation_type: type[AggregatedEvaluation]
    ) -> Optional[AggregationOverview[AggregatedEvaluation]]:
        return self._aggregation_overviews.get(aggregation_id, None)
