from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

from intelligence_layer.evaluation.aggregation.domain import (
    AggregatedEvaluation,
    AggregationOverview,
)
from intelligence_layer.evaluation.infrastructure.file_system_based_repository import (
    FileSystemBasedRepository,
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

        Returns:
            An :class:`Iterable` of :class:`AggregationOverview`s.
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


class FileSystemAggregationRepository(AggregationRepository, FileSystemBasedRepository):
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

        if not self.exists(file_path):
            return None

        content = self.read_utf8(file_path)
        return AggregationOverview[aggregation_type].model_validate_json(  # type:ignore
            content
        )

    def aggregation_overview_ids(self) -> Sequence[str]:
        return sorted(
            [
                Path(f["name"]).stem
                for f in self._file_system.ls(
                    self.path_to_str(self._aggregation_root_directory()), detail=True
                )
                if isinstance(f, Dict) and Path(f["name"]).suffix == ".json"
            ]
        )

    def _aggregation_root_directory(self) -> Path:
        path = self._root_directory / "aggregations"
        path.mkdir(exist_ok=True)
        return path

    def _aggregation_directory(self, evaluation_id: str) -> Path:
        path = self._aggregation_root_directory() / evaluation_id
        path.mkdir(exist_ok=True)
        return path

    def _aggregation_overview_path(self, aggregation_id: str) -> Path:
        return self._aggregation_directory(aggregation_id).with_suffix(".json")
