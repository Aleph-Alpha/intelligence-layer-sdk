from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Optional
from unittest.mock import patch
from uuid import uuid4

import pytest
from fsspec import AbstractFileSystem  # type: ignore
from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.core.task import Input
from intelligence_layer.evaluation import Dataset, Example, HuggingFaceDatasetRepository
from intelligence_layer.evaluation.dataset.domain import ExpectedOutput


class HuggingFaceDatasetRepositoryTestWrapper(HuggingFaceDatasetRepository):
    def __init__(
        self, repository_id: str, token: str, private: bool, caching: bool = True
    ) -> None:
        super().__init__(repository_id, token, private, caching)
        self.counter = 0

    def examples(
        self,
        dataset_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
        examples_to_skip: Optional[frozenset[str]] = None,
    ) -> Iterable[Example[Input, ExpectedOutput]]:
        self.counter += 1
        return super().examples(
            dataset_id, input_type, expected_output_type, examples_to_skip
        )


@fixture
def cached_mocked_hugging_face_dataset_wrapper(
    temp_file_system: AbstractFileSystem,
) -> HuggingFaceDatasetRepository:
    class_to_patch = "intelligence_layer.evaluation.dataset.hugging_face_dataset_repository.HuggingFaceDatasetRepository"
    with (
        patch(f"{class_to_patch}.create_repository", autospec=True),
        patch(
            f"{class_to_patch}.delete_repository",
            autospec=True,
        ),
    ):
        repo = HuggingFaceDatasetRepositoryTestWrapper(
            repository_id="doesn't-matter",
            token="non-existing-token",
            private=True,
            caching=True,
        )
        repo._file_system = temp_file_system
        return repo


def test_opens_files_only_once_when_reading_multiple_examples(
    cached_mocked_hugging_face_dataset_wrapper: HuggingFaceDatasetRepositoryTestWrapper,
) -> None:
    dataset = cached_mocked_hugging_face_dataset_wrapper.create_dataset([], "temp")

    cached_mocked_hugging_face_dataset_wrapper.example(dataset.id, "", str, str)
    cached_mocked_hugging_face_dataset_wrapper.example(dataset.id, "", str, str)

    assert cached_mocked_hugging_face_dataset_wrapper.counter == 1


def test_forgets_datasets_after_deleting_one(
    cached_mocked_hugging_face_dataset_wrapper: HuggingFaceDatasetRepositoryTestWrapper,
) -> None:
    dataset = cached_mocked_hugging_face_dataset_wrapper.create_dataset([], "temp")

    cached_mocked_hugging_face_dataset_wrapper.example(dataset.id, "", str, str)
    cached_mocked_hugging_face_dataset_wrapper.delete_dataset(dataset.id)

    with pytest.raises(ValueError):
        cached_mocked_hugging_face_dataset_wrapper.examples(dataset.id, str, str)


class DummyAggregatedEvaluation(BaseModel):
    score: float


# these fixtures should only be used once and are here for readable tests
# because creating/deleting HuggingFace repositories can be rate-limited
@fixture(scope="session")
def hugging_face_dataset_repository(
    hugging_face_test_repository_id: str, hugging_face_token: str
) -> Iterable[HuggingFaceDatasetRepository]:
    repo = HuggingFaceDatasetRepository(
        repository_id=hugging_face_test_repository_id,
        token=hugging_face_token,
        private=True,
    )
    try:
        yield repo
    finally:
        repo.delete_repository()


@fixture(scope="session")
def hugging_face_repository_with_dataset_and_examples(
    hugging_face_dataset_repository: HuggingFaceDatasetRepository,
) -> Iterable[
    tuple[HuggingFaceDatasetRepository, Dataset, Sequence[Example[str, str]]]
]:
    examples = [
        Example(input="hey", expected_output="hey hey"),
        Example(input="hi", expected_output="hi hi"),
    ]
    id = str(uuid4())
    try:
        dataset = hugging_face_dataset_repository.create_dataset(
            examples=examples, dataset_name="test-hg-dataset", id=id
        )
    except Exception as e:
        hugging_face_dataset_repository.delete_dataset(id)
        raise e

    try:
        yield hugging_face_dataset_repository, dataset, examples
    finally:
        hugging_face_dataset_repository.delete_dataset(id)


def test_hugging_face_repository_supports_all_operations_for_created_dataset(
    hugging_face_repository_with_dataset_and_examples: tuple[
        HuggingFaceDatasetRepository, Dataset, Sequence[Example[str, str]]
    ],
) -> None:
    (hugging_face_repository, dataset, examples) = (
        hugging_face_repository_with_dataset_and_examples
    )

    # created dataset is stored
    stored_dataset = hugging_face_repository.dataset(dataset.id)
    assert stored_dataset == dataset
    # datasets are stored in a subdirectory on huggingface
    path_to_file = Path(
        f"datasets/{hugging_face_repository._repository_id}/datasets/{dataset.id}/{dataset.id}.json"
    )
    assert hugging_face_repository.exists(path_to_file)

    stored_dataset_ids = list(hugging_face_repository.dataset_ids())
    # non-dataset files are not retrieved as datasets
    assert ".gitattributes" not in stored_dataset_ids
    assert "README.md" not in stored_dataset_ids
    # created dataset is included
    assert dataset.id in stored_dataset_ids

    # given examples are stored and can be accessed via their ID
    assert list(hugging_face_repository.examples(dataset.id, str, str)) == sorted(
        examples, key=lambda e: e.id
    )
    for example in examples:
        assert (
            hugging_face_repository.example(dataset.id, example.id, str, str) == example
        )

    # one gets None for a not existing example ID
    assert (
        hugging_face_repository.example(dataset.id, "not-existing-example-id", str, str)
        is None
    )

    # deleting an existing dataset works
    hugging_face_repository.delete_dataset(dataset.id)
    with pytest.raises(ValueError):
        hugging_face_repository.examples(dataset.id, str, str)
    assert hugging_face_repository.dataset(dataset.id) is None
