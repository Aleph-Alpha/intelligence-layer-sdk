from pathlib import Path
from typing import Iterable, Sequence, Tuple
from uuid import uuid4

import pytest
from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.evaluation import Dataset, Example, HuggingFaceDatasetRepository


class DummyAggregatedEvaluation(BaseModel):
    score: float


@fixture(scope="session")
def hugging_face_dataset_repository_id() -> str:
    return f"Aleph-Alpha/test-datasets-{str(uuid4())}"


# these fixtures should only be used once and are here for readable tests
@fixture(scope="session")
def hugging_face_dataset_repository(
    hugging_face_dataset_repository_id: str, hugging_face_token: str
) -> Iterable[HuggingFaceDatasetRepository]:
    # this repository should already exist and does not have to be deleted after the tests
    repo = HuggingFaceDatasetRepository(
        repository_id=hugging_face_dataset_repository_id,
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
    Tuple[HuggingFaceDatasetRepository, Dataset, Sequence[Example[str, str]]]
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
    hugging_face_repository_with_dataset_and_examples: Tuple[
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
