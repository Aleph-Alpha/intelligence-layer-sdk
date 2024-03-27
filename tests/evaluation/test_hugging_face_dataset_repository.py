from pathlib import Path
from typing import Iterable, Sequence, Tuple
from uuid import uuid4

import huggingface_hub  # type: ignore
import pytest
from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.evaluation import Dataset, Example, HuggingFaceDatasetRepository


class DummyAggregatedEvaluation(BaseModel):
    score: float


@fixture(scope="session")
def hugging_face_dataset_repository_id() -> str:
    return "Aleph-Alpha/test-datasets"


@fixture(scope="session")
def hugging_face_dataset_repository(
    hugging_face_dataset_repository_id: str, hugging_face_token: str
) -> HuggingFaceDatasetRepository:
    # this repository should already exist and does not have to be deleted after the tests
    return HuggingFaceDatasetRepository(
        repository_id=hugging_face_dataset_repository_id,
        token=hugging_face_token,
        private=True,
    )


@fixture
def example_1() -> Example[str, str]:
    return Example(input="hey", expected_output="hey hey")


@fixture
def example_2() -> Example[str, str]:
    return Example(input="hi", expected_output="hi hi")


@fixture
def hugging_face_repository_with_dataset_and_examples(
    hugging_face_dataset_repository: HuggingFaceDatasetRepository,
    example_1: Example[str, str],
    example_2: Example[str, str],
) -> Iterable[
    Tuple[HuggingFaceDatasetRepository, Dataset, Sequence[Example[str, str]]]
]:
    examples = [example_1, example_2]
    dataset = hugging_face_dataset_repository.create_dataset(
        examples=examples, dataset_name="test-hg-dataset"
    )

    try:
        yield hugging_face_dataset_repository, dataset, examples
    finally:
        hugging_face_dataset_repository.delete_dataset(dataset.id)


def test_hugging_face_repository_can_create_and_delete_a_repository(
    hugging_face_token: str,
) -> None:
    repository_id = f"Aleph-Alpha/test-{uuid4()}"

    assert not huggingface_hub.repo_exists(
        repo_id=repository_id,
        token=hugging_face_token,
        repo_type="dataset",
    ), f"This is very unlikely but it seems that the repository with the ID {repository_id} already exists."

    created_repository = HuggingFaceDatasetRepository(
        repository_id=repository_id,
        token=hugging_face_token,
        private=True,
    )

    try:
        assert huggingface_hub.repo_exists(
            repo_id=repository_id,
            token=hugging_face_token,
            repo_type="dataset",
        )
        created_repository.delete_repository()
        assert not huggingface_hub.repo_exists(
            repo_id=repository_id,
            token=hugging_face_token,
            repo_type="dataset",
        )
    finally:
        huggingface_hub.delete_repo(
            repo_id=repository_id,
            token=hugging_face_token,
            repo_type="dataset",
            missing_ok=True,
        )


def test_examples_returns_an_empty_list_for_not_existing_dataset_id(
    hugging_face_dataset_repository: HuggingFaceDatasetRepository,
) -> None:
    with pytest.raises(ValueError):
        hugging_face_dataset_repository.examples("not-existing-dataset-id", str, str)


def test_example_returns_none_for_not_existing_ids(
    hugging_face_dataset_repository: HuggingFaceDatasetRepository,
) -> None:
    with pytest.raises(ValueError):
        hugging_face_dataset_repository.example(
            "not-existing-dataset-id", "not-existing-example-id", str, str
        )


def test_delete_dataset_does_not_fail_for_not_existing_dataset_id(
    hugging_face_dataset_repository: HuggingFaceDatasetRepository,
) -> None:
    try:
        hugging_face_dataset_repository.delete_dataset("not-existing-dataset-id")
    except Exception:
        assert False, "Deleting a not-existing dataset should not throw an exception"


def test_hugging_face_repository_supports_all_operations_for_created_dataset(
    hugging_face_repository_with_dataset_and_examples: Tuple[
        HuggingFaceDatasetRepository, Dataset, Sequence[Example[str, str]]
    ]
) -> None:
    (hugging_face_repository_, dataset, examples) = (
        hugging_face_repository_with_dataset_and_examples
    )

    # created dataset is stored
    stored_dataset = hugging_face_repository_.dataset(dataset.id)
    assert stored_dataset == dataset

    stored_dataset_ids = list(hugging_face_repository_.dataset_ids())
    # non-dataset files are not retrieved as datasets
    assert ".gitattributes" not in stored_dataset_ids
    assert "README.md" not in stored_dataset_ids
    # created dataset is included
    assert dataset.id in stored_dataset_ids

    # given examples are stored and can be accessed via their ID
    assert list(hugging_face_repository_.examples(dataset.id, str, str)) == sorted(
        examples, key=lambda e: e.id
    )
    for example in examples:
        assert (
            hugging_face_repository_.example(dataset.id, example.id, str, str)
            == example
        )

    # one gets None for a not existing example ID
    assert (
        hugging_face_repository_.example(
            dataset.id, "not-existing-example-id", str, str
        )
        is None
    )

    # deleting an existing dataset works
    hugging_face_repository_.delete_dataset(dataset.id)
    with pytest.raises(ValueError):
        hugging_face_repository_.examples(dataset.id, str, str)
    assert hugging_face_repository_.dataset(dataset.id) is None


def test_file_exists_in_subdirectory(
    hugging_face_repository_with_dataset_and_examples: Tuple[
        HuggingFaceDatasetRepository, Dataset, Sequence[Example[str, str]]
    ]
) -> None:
    (hugging_face_repository, dataset, examples) = (
        hugging_face_repository_with_dataset_and_examples
    )
    path_to_file = Path(
        f"datasets/{hugging_face_repository._repository_id}/datasets/{dataset.id}/{dataset.id}.json"
    )

    file_exists = hugging_face_repository.exists(path_to_file)
    assert file_exists
