from pathlib import Path

import huggingface_hub  # type: ignore
from huggingface_hub import HfFileSystem, create_repo, get_paths_info

from intelligence_layer.evaluation.aggregation.file_aggregation_repository import (
    FileSystemAggregationRepository,
)
from intelligence_layer.evaluation.infrastructure.hugging_face_repository import (
    HuggingFaceRepository,
)


class HuggingFaceAggregationRepository(
    FileSystemAggregationRepository, HuggingFaceRepository
):

    def __init__(self, repository_id: str, token: str, private: bool) -> None:
        assert repository_id[-1] != "/"

        create_repo(
            repo_id=repository_id,
            token=token,
            repo_type=HuggingFaceRepository._REPO_TYPE,
            private=private,
            exist_ok=True,
        )

        file_system = HfFileSystem(token=token)
        root_directory = Path(
            f"{HuggingFaceRepository._ROOT_DIRECTORY_PREFIX_}/{repository_id}"
        )
        super().__init__(file_system, root_directory)

        self._repository_id = repository_id
        self._file_system = file_system  # for better type checks

    def delete_repository(self) -> None:
        huggingface_hub.delete_repo(
            repo_id=self._repository_id,
            token=self._file_system.token,
            repo_type=HuggingFaceRepository._REPO_TYPE,
            missing_ok=True,
        )

    # The `exists` function implemented for the Hugginface file system (HfFileSystem)
    # cannot find files that are nested in folders, but only top-level files in the repository.
    # Here, we overwrite the method defined in the `AbstractFileSystem`.
    # This fix will have to be implemented in `HuggingFaceRepository` and/or `HuggingFaceDatasetRepository`.
    def exists(self, path: Path) -> bool:
        try:
            path_relative_to_repository_id = path.relative_to(self._repository_id)
        except ValueError:
            return False

        return (
            len(
                get_paths_info(
                    self._repository_id,
                    self.path_to_str(path_relative_to_repository_id),
                    repo_type="dataset",
                    token=self._file_system.token,
                )
            )
            != 0
        )
