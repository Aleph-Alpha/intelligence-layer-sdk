from pathlib import Path

from fsspec.implementations.local import LocalFileSystem  # type: ignore

from intelligence_layer.evaluation.aggregation.aggregation_repository import (
    FileSystemAggregationRepository,
)


class FileAggregationRepository(FileSystemAggregationRepository):
    def __init__(self, root_directory: Path) -> None:
        super().__init__(LocalFileSystem(), root_directory)

    @staticmethod
    def path_to_str(path: Path) -> str:
        return str(path)
