from typing import Optional


from intelligence_layer.evaluation.dataset.domain import Dataset
from intelligence_layer.evaluation.dataset.file_dataset_repository import (
    FileSystemDatasetRepository,
)
from intelligence_layer.evaluation.infrastructure.hugging_face_repository import (
    HuggingFaceRepository,
)


class HuggingFaceDatasetRepository(HuggingFaceRepository, FileSystemDatasetRepository):
    def delete_dataset(self, dataset_id: str) -> None:
        """Deletes a dataset identified by the given dataset ID.

        This implementation should be backwards compatible to datasets
        created without a dataset object (i.e., there is no dataset file
        with dataset metadata).

        Note, that HuggingFace API does not seem to support deleting not-existing files.

        Args:
            dataset_id: Dataset ID of the dataset to delete.
        """
        if self.exists(self._dataset_examples_path(dataset_id)):
            self._file_system.rm(
                self.path_to_str(self._dataset_examples_path(dataset_id))
            )

        if self.exists(self._dataset_path(dataset_id)):
            self._file_system.rm(self.path_to_str(self._dataset_path(dataset_id)))

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
        if not self.exists(dataset_file_path):
            if not self.exists(examples_file_path):
                return None
            else:
                return Dataset(id=dataset_id, name=f"HuggingFace dataset {dataset_id}")

        return super().dataset(dataset_id)
