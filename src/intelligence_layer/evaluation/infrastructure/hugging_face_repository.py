from pathlib import Path

from intelligence_layer.evaluation.dataset.file_dataset_repository import (
    FileSystemDatasetRepository,
)


class HuggingFaceRepository(FileSystemDatasetRepository):
    """HuggingFace base repository"""

    _REPO_TYPE = "dataset"
    _ROOT_DIRECTORY_PREFIX_ = "datasets"  # HuggingFace API root directory

    @staticmethod
    def path_to_str(path: Path) -> str:
        return path.as_posix()
