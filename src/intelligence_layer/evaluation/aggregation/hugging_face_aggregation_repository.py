from pathlib import Path

import huggingface_hub  # type: ignore
from huggingface_hub import HfFileSystem, create_repo

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
