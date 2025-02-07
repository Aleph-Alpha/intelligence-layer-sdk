from pathlib import Path

import huggingface_hub

from intelligence_layer.evaluation.infrastructure.file_system_based_repository import (
    FileSystemBasedRepository,
)


class HuggingFaceRepository(FileSystemBasedRepository):
    """HuggingFace base repository."""

    _REPO_TYPE = "dataset"
    _ROOT_DIRECTORY_PREFIX_ = "datasets"  # HuggingFace API root directory

    @staticmethod
    def path_to_str(path: Path) -> str:
        return path.as_posix()

    def __init__(self, repository_id: str, token: str, private: bool) -> None:
        """Create a HuggingFace repository.

        Creates a corresponding repository and initializes the file system.

        Args:
            repository_id: The HuggingFace namespace and repository name, separated by a "/".
            token: The HuggingFace authentication token.
            private: Whether the dataset repository should be private.
        """
        assert repository_id[-1] != "/"
        self.create_repository(repository_id, token, private)

        file_system = huggingface_hub.HfFileSystem(token=token)
        root_directory = Path(f"{self._ROOT_DIRECTORY_PREFIX_}/{repository_id}")

        super().__init__(file_system, root_directory)
        self._repository_id = repository_id
        # the file system is assigned in super init but this fixes the typing
        self._file_system: huggingface_hub.HfFileSystem

    def create_repository(self, repository_id: str, token: str, private: bool) -> None:
        huggingface_hub.create_repo(
            repo_id=repository_id,
            token=token,
            repo_type=self._REPO_TYPE,
            private=private,
            exist_ok=True,
        )

    def delete_repository(self) -> None:
        huggingface_hub.delete_repo(
            repo_id=self._repository_id,
            token=self._file_system.token,
            repo_type=self._REPO_TYPE,
            missing_ok=True,
        )
