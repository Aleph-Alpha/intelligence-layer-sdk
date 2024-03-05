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


def requires_token() -> MarkDecorator:
    return pytest.mark.skipif(
        "HUGGING_FACE_TOKEN" in os.environ.keys(),
        reason="HUGGING_FACE_TOKEN not set, necessary for for current test",
    )


@requires_token()
def test_hf_database_non_existing(hf_repository: HuggingFaceDatasetRepository) -> None:
    assert hf_repository.examples("lol", str, str) == []
    assert hf_repository.example("lol", "abc", str, str) is None
    hf_repository.delete_dataset("lol")
    # make sure random files are not actually datasets
    datasets = list(hf_repository.dataset_ids())
    assert ".gitattributes" not in datasets
    assert "README.md" not in datasets


@requires_token()
def test_hf_database_operations(
    hf_repository: HuggingFaceDatasetRepository,
    example1: Example[str, str],
    example2: Example[str, str],
) -> None:
    original_examples = [example1, example2]
    dataset_id = hf_repository.create_dataset(original_examples)
    try:
        assert dataset_id in list(hf_repository.dataset_ids())
        examples = hf_repository.examples(dataset_id, str, str)
        assert examples is not None
        assert list(examples) == original_examples
        assert hf_repository.example(dataset_id, example1.id, str, str) == example1
        assert hf_repository.example(dataset_id, "abc", str, str) is None
        hf_repository.delete_dataset(dataset_id)
        assert hf_repository.examples(dataset_id, str, str) is None
    finally:
        hf_repository.delete_dataset(dataset_id)
