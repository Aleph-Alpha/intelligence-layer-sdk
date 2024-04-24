from pathlib import Path
from typing import Iterable

import pytest
from fsspec import AbstractFileSystem  # type: ignore
from pytest import fixture

from intelligence_layer.core.task import Input
from intelligence_layer.evaluation.dataset.domain import Example, ExpectedOutput
from intelligence_layer.evaluation.dataset.file_dataset_repository import (
    FileSystemDatasetRepository,
)


class FileDatasetRepositoryTestWrapper(FileSystemDatasetRepository):
    def __init__(
        self, filesystem: AbstractFileSystem, root_directory: Path, caching: bool
    ) -> None:
        super().__init__(filesystem, root_directory, caching)
        self.counter = 0

    def examples(
        self,
        dataset_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
    ) -> Iterable[Example[Input, ExpectedOutput]]:
        self.counter += 1
        return super().examples(dataset_id, input_type, expected_output_type)


@fixture
def file_data_repo_stub(
    temp_file_system: AbstractFileSystem,
) -> FileDatasetRepositoryTestWrapper:
    return FileDatasetRepositoryTestWrapper(
        temp_file_system, Path("Root"), caching=True
    )


def test_opens_files_only_once_when_reading_multiple_examples(
    file_data_repo_stub: FileDatasetRepositoryTestWrapper,
) -> None:
    dataset = file_data_repo_stub.create_dataset([], "temp")

    file_data_repo_stub.example(dataset.id, "", str, str)
    file_data_repo_stub.example(dataset.id, "", str, str)

    assert file_data_repo_stub.counter == 1


def test_forgets_datasets_after_deleting_one(
    file_data_repo_stub: FileDatasetRepositoryTestWrapper,
) -> None:
    dataset = file_data_repo_stub.create_dataset([], "temp")

    file_data_repo_stub.example(dataset.id, "", str, str)
    file_data_repo_stub.delete_dataset(dataset.id)

    with pytest.raises(ValueError):
        file_data_repo_stub.examples(dataset.id, str, str)
