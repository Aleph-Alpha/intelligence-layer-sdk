from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

from intelligence_layer.evaluation.data_storage.utils import FileBasedRepository
from intelligence_layer.evaluation.domain import (
    AggregatedEvaluation,
    AggregationOverview,
)


class AggregationRepository(ABC):
    @abstractmethod
    def aggregation_overview(
        self, id: str, stat_type: type[AggregatedEvaluation]
    ) -> AggregationOverview[AggregatedEvaluation] | None:
        """Returns all failed :class:`ExampleResult` instances of a given run

        Args:
            id: Identifier of the TODO
            stat_type:


        Returns:
            :class:`EvaluationOverview` if one was found, `None` otherwise.
        """
        ...

    @abstractmethod
    def store_aggregation_overview(
        self, overview: AggregationOverview[AggregatedEvaluation]
    ) -> None:
        """Stores an :class:`AggregationOverview` in the repository.

        Args:
            overview: The overview to be persisted.
        """
        ...


class FileAggregationRepository(AggregationRepository, FileBasedRepository):
    def _aggregation_root_directory(self) -> Path:
        path = self._root_directory / "aggregation"
        path.mkdir(exist_ok=True)
        return path

    def _aggregation_overview_path(self, id: str) -> Path:
        return (self._aggregation_root_directory() / id).with_suffix(".json")

    def aggregation_overview(
        self, id: str, stat_type: type[AggregatedEvaluation]
    ) -> AggregationOverview[AggregatedEvaluation] | None:
        file_path = self._aggregation_overview_path(id)
        if not file_path.exists():
            return None
        content = self.read_utf8(file_path)
        return AggregationOverview[stat_type].model_validate_json(  # type:ignore
            content
        )

    def store_aggregation_overview(
        self, overview: AggregationOverview[AggregatedEvaluation]
    ) -> None:
        self.write_utf8(
            self._aggregation_overview_path(overview.id),
            overview.model_dump_json(indent=2),
        )

    def aggregation_ids(self) -> Sequence[str]:
        return [path.stem for path in self._aggregation_root_directory().glob("*.json")]


class InMemoryAggregationRepository(AggregationRepository):
    def __init__(self) -> None:
        super().__init__()
        self._aggregation_overviews: dict[str, AggregationOverview[Any]] = dict()

    def aggregation_overview(
        self, id: str, stat_type: type[AggregatedEvaluation]
    ) -> AggregationOverview[AggregatedEvaluation] | None:
        return self._aggregation_overviews[id]

    def store_aggregation_overview(
        self, overview: AggregationOverview[AggregatedEvaluation]
    ) -> None:
        self._aggregation_overviews[overview.id] = overview
