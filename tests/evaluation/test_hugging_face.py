import os

import pytest
from dotenv import load_dotenv
from pytest import MarkDecorator, fixture

from intelligence_layer.evaluation import Example, HuggingFaceDatasetRepository


@fixture(scope="session")
def hf_token() -> str:
    load_dotenv()
    token = os.getenv("HUGGING_FACE_TOKEN")
    assert isinstance(token, str)
    return token


@fixture(scope="session")
def hf_repository(hf_token: str) -> HuggingFaceDatasetRepository:
    return HuggingFaceDatasetRepository(
        "Aleph-Alpha/test-datasets", token=hf_token, private=True
    )


@fixture
def example1() -> Example[str, str]:
    return Example(input="hey", expected_output="ho", id="0")


@fixture
def example2() -> Example[str, str]:
    return Example(input="ho", expected_output="hey", id="1")


def skip_if_required_token_not_set() -> MarkDecorator:
    return pytest.mark.skipif(
        "HUGGING_FACE_TOKEN" in os.environ.keys(),
        reason="HUGGING_FACE_TOKEN not set, necessary for for current test",
    )


@skip_if_required_token_not_set()
def test_hf_database_non_existing(hf_repository: HuggingFaceDatasetRepository) -> None:
    # not existing IDs
    assert hf_repository.examples("lol", str, str) == []
    assert hf_repository.example("lol", "abc", str, str) is None

    # deleting a not-existing dataset
    try:
        hf_repository.delete_dataset("lol")
    except Exception:
        assert False, "Deleting a not-existing dataset should not throw an exception"

    # non-dataset files are not retrieved as datasets
    datasets = list(hf_repository.dataset_ids())

    assert ".gitattributes" not in datasets
    assert "README.md" not in datasets


@skip_if_required_token_not_set()
def test_hf_database_operations(
    hf_repository: HuggingFaceDatasetRepository,
    example1: Example[str, str],
    example2: Example[str, str],
) -> None:
    examples = [example1, example2]

    dataset_id = hf_repository.create_dataset(examples, "test-hg-dataset").id

    try:
        stored_dataset_ids = list(hf_repository.dataset_ids())

        # non-dataset files are not retrieved as datasets
        assert ".gitattributes" not in stored_dataset_ids
        assert "README.md" not in stored_dataset_ids

        # created dataset is stored
        assert dataset_id in stored_dataset_ids

        # given examples are stored and can be accessed via their ID
        assert list(hf_repository.examples(dataset_id, str, str)) == examples
        for example in examples:
            assert hf_repository.example(dataset_id, example.id, str, str) == example

        # example() with not-existing example ID returns None
        assert hf_repository.example(dataset_id, "abc", str, str) is None

        # deleting a dataset works
        hf_repository.delete_dataset(dataset_id)

        assert hf_repository.examples(dataset_id, str, str) == []
    finally:
        hf_repository.delete_dataset(dataset_id)
