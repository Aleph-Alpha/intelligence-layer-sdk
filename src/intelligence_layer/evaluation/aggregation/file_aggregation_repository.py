from pathlib import Path
from typing import Dict, Optional, Sequence

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


class FileAggregationRepository(FileSystemAggregationRepository):
    def __init__(self, root_directory: Path) -> None:
        super().__init__(LocalFileSystem(), root_directory)

    @staticmethod
    def path_to_str(path: Path) -> str:
        return str(path)
