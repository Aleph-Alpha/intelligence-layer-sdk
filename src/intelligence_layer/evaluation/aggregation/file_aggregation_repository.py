from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from fsspec.implementations.local import LocalFileSystem  # type: ignore

from intelligence_layer.evaluation.aggregation.aggregation_repository import (
    AggregationRepository,
)
from intelligence_layer.evaluation.aggregation.domain import (
    AggregatedEvaluation,
    AggregationOverview,
)
from intelligence_layer.evaluation.infrastructure.file_system_based_repository import (
    FileSystemBasedRepository,
)


class FileSystemAggregationRepository(AggregationRepository, FileSystemBasedRepository):
    _SUB_DIRECTORY = "aggregations"

    def store_aggregation_overview(
        self, aggregation_overview: AggregationOverview[AggregatedEvaluation]
    ) -> None:
        self.write_utf8(
            self._aggregation_overview_path(aggregation_overview.id),
            aggregation_overview.model_dump_json(indent=2),
            create_parents=True,
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
        return sorted(self.file_names(self._aggregation_root_directory()))

    def _aggregation_root_directory(self) -> Path:
        return self._root_directory / self._SUB_DIRECTORY

    def _aggregation_directory(self, evaluation_id: str) -> Path:
        return self._aggregation_root_directory() / evaluation_id

    def _aggregation_overview_path(self, aggregation_id: str) -> Path:
        return self._aggregation_directory(aggregation_id).with_suffix(".json")


class FileAggregationRepository(FileSystemAggregationRepository):
    def __init__(self, root_directory: Path) -> None:
        super().__init__(LocalFileSystem(), root_directory)

    @staticmethod
    def path_to_str(path: Path) -> str:
        return str(path)
