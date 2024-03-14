from pathlib import Path

import huggingface_hub  # type: ignore
from huggingface_hub import HfFileSystem, create_repo

from intelligence_layer.evaluation.data_storage.aggregation_repository import (
    FileSystemAggregationRepository,
)
from intelligence_layer.evaluation.data_storage.dataset_repository import (
    FileSystemDatasetRepository,
)


class HuggingFaceDatasetRepository(FileSystemDatasetRepository):
    _REPO_TYPE = "dataset"

    def __init__(self, database_name: str, token: str, private: bool) -> None:
        assert database_name[-1] != "/"
        create_repo(
            database_name,
            token=token,
            repo_type=HuggingFaceDatasetRepository._REPO_TYPE,
            exist_ok=True,
            private=private,
        )
        self._database_name = database_name
        super().__init__(HfFileSystem(token=token), Path(f"datasets/{database_name}"))

    def delete_repository(self) -> None:
        huggingface_hub.delete_repo(
            database_name=self._database_name,
            token=self._file_system.token,
            repo_type=HuggingFaceDatasetRepository._REPO_TYPE,
            missing_ok=True,
        )


class HuggingFaceAggregationRepository(FileSystemAggregationRepository):
    _REPO_TYPE = "dataset"

    def __init__(self, database_name: str, token: str, private: bool) -> None:
        assert database_name[-1] != "/"
        create_repo(
            database_name,
            token=token,
            repo_type=HuggingFaceAggregationRepository._REPO_TYPE,
            exist_ok=True,
            private=private,
        )
        self._database_name = database_name
        fs = HfFileSystem(token=token)
        root_directory = Path(f"datasets/{database_name}")
        super().__init__(fs, root_directory)

    def delete_repository(self) -> None:
        huggingface_hub.delete_repo(
            repo_id=self._database_name,
            token=self._fs.token,
            repo_type=HuggingFaceAggregationRepository._REPO_TYPE,
            missing_ok=True,
        )

    @staticmethod
    def path_to_str(path: Path) -> str:
        return path.as_posix()
