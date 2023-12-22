import os

from dotenv import load_dotenv
from pytest import fixture

from intelligence_layer.core import Example
from intelligence_layer.core.evaluation.hugging_face import HuggingFaceDatasetRepository


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
def example() -> Example[str, str]:
    return Example(input="hey", expected_output="ho")


def test_hf_database_non_existing(hf_repository: HuggingFaceDatasetRepository) -> None:
    assert hf_repository.examples_by_id("lol", str, str) is None
    assert hf_repository.example("lol", "abc", str, str) is None
    hf_repository.delete_dataset("lol")
    # make sure random files are not actually datasets
    datasets = list(hf_repository.list_datasets())
    assert ".gitattributes" not in datasets
    assert "README.md" not in datasets


def test_hf_database_operations(
    hf_repository: HuggingFaceDatasetRepository, example: Example[str, str]
) -> None:
    dataset_id = hf_repository.create_dataset([example])
    try:
        assert dataset_id in list(hf_repository.list_datasets())
        examples = hf_repository.examples_by_id(dataset_id, str, str)
        assert examples is not None
        assert [e for e in examples] == [example]
        assert hf_repository.example(dataset_id, example.id, str, str) == example
        assert hf_repository.example(dataset_id, "abc", str, str) is None
        hf_repository.delete_dataset(dataset_id)
        assert hf_repository.examples_by_id(dataset_id, str, str) is None
    finally:
        hf_repository.delete_dataset(dataset_id)
