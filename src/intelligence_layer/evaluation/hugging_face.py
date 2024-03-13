from pathlib import Path
from typing import Optional

import huggingface_hub  # type: ignore
from huggingface_hub import HfFileSystem, create_repo

from intelligence_layer.evaluation.data_storage.aggregation_repository import (
    FileSystemAggregationRepository,
)
from intelligence_layer.evaluation.data_storage.dataset_repository import (
    FileSystemDatasetRepository,
)
from intelligence_layer.evaluation.domain import Dataset


class HuggingFaceDatasetRepository(FileSystemDatasetRepository):
    _REPO_TYPE = "dataset"
    _ROOT_DIRECTORY_PREFIX_ = "datasets"  # HuggingFace API root directory

    def __init__(self, repository_id: str, token: str, private: bool) -> None:
        """Create a HuggingFace dataset repository

        Args:
            repository_id: The HuggingFace namespace and repository name separated by a "/".
            token: The HuggingFace token.
            private: Whether the dataset repository should be private.
        """
        assert repository_id[-1] != "/"

        create_repo(
            repo_id=repository_id,
            token=token,
            repo_type=HuggingFaceDatasetRepository._REPO_TYPE,
            private=private,
            exist_ok=True,
        )

        file_system = HfFileSystem(token=token)
        root_directory = Path(
            f"{HuggingFaceDatasetRepository._ROOT_DIRECTORY_PREFIX_}/{repository_id}"
        )
        super().__init__(file_system, root_directory)

        self._repository_id = repository_id
        self._file_system = file_system  # for better type checks

    def delete_repository(self) -> None:
        huggingface_hub.delete_repo(
            repo_id=self._repository_id,
            token=self._file_system.token,
            repo_type=HuggingFaceDatasetRepository._REPO_TYPE,
            missing_ok=True,
        )

    def delete_dataset(self, dataset_id: str) -> None:
        """Deletes a dataset identified by the given dataset ID.

        This implementation should be backwards compatible to datasets
        created without a dataset object (i.e., there is no dataset file
        with dataset metadata).

        Note, that HuggingFace API does not seem to support deleting not-existing files.

        Args:
            dataset_id: Dataset ID of the dataset to delete.
        """
        if self._file_system.exists(str(self._dataset_examples_path(dataset_id))):
            self._file_system.rm(str(self._dataset_examples_path(dataset_id)))

        if self._file_system.exists(str(self._dataset_path(dataset_id))):
            self._file_system.rm(str(self._dataset_path(dataset_id)))

    def dataset(self, dataset_id: str) -> Optional[Dataset]:
        """Returns a dataset identified by the given dataset ID.

        This implementation should be backwards compatible to datasets
        created without a dataset object (i.e., there is no dataset file
        with dataset metadata).

        Args:
            dataset_id: Dataset ID of the dataset to delete.

        Returns:
            :class:`Dataset` if it was not, `None` otherwise.
        """
        dataset_file_path = self._dataset_path(dataset_id)
        examples_file_path = self._dataset_examples_path(dataset_id)
        if not self._file_system.exists(str(dataset_file_path)):
            if not self._file_system.exists(str(examples_file_path)):
                return None
            else:
                return Dataset(id=dataset_id, name=f"HuggingFace dataset {dataset_id}")

        return super().dataset(dataset_id)

    def _dataset_root_directory(self) -> Path:
        # we override this method as the existing HuggingFace datasets have a different file structure
        return self._root_directory

    def _dataset_directory(self, dataset_id: str) -> Path:
        # we override this method as the existing HuggingFace datasets have a different file structure
        return self._root_directory


class HuggingFaceAggregationRepository(FileSystemAggregationRepository):
    _REPO_TYPE = "dataset"
    _ROOT_DIRECTORY_PREFIX_ = "datasets"  # HuggingFace API root directory

    def __init__(self, repository_id: str, token: str, private: bool) -> None:
        assert repository_id[-1] != "/"

        create_repo(
            repo_id=repository_id,
            token=token,
            repo_type=HuggingFaceAggregationRepository._REPO_TYPE,
            private=private,
            exist_ok=True,
        )

        file_system = HfFileSystem(token=token)
        root_directory = Path(
            f"{HuggingFaceAggregationRepository._ROOT_DIRECTORY_PREFIX_}/{repository_id}"
        )
        super().__init__(file_system, root_directory)

        self._repository_id = repository_id
        self._file_system = file_system  # for better type checks

    def delete_repository(self) -> None:
        huggingface_hub.delete_repo(
            repo_id=self._repository_id,
            token=self._file_system.token,
            repo_type=HuggingFaceAggregationRepository._REPO_TYPE,
            missing_ok=True,
        )

    @staticmethod
    def path_to_str(path: Path) -> str:
        return path.as_posix()
