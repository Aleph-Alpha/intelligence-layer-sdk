from pathlib import Path
from typing import Iterable

from pytest import FixtureRequest, fixture, mark

from intelligence_layer.evaluation import (
    DatasetRepository,
    Example,
    FileDatasetRepository,
)
from tests.conftest import DummyStringInput, DummyStringOutput


@fixture
def file_dataset_repository(tmp_path: Path) -> FileDatasetRepository:
    return FileDatasetRepository(tmp_path)


test_repository_fixtures = [
    "file_dataset_repository",
    "in_memory_dataset_repository",
]


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_file_dataset_repository_can_store_dataset(
    repository_fixture: str,
    dummy_string_example: Example[DummyStringInput, DummyStringOutput],
    request: FixtureRequest,
) -> None:
    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)
    dataset_id = dataset_repository.create_dataset(examples=[dummy_string_example])
    examples = dataset_repository.examples(
        dataset_id,
        input_type=DummyStringInput,
        expected_output_type=DummyStringOutput,
    )
    assert examples is not None
    examples = list(examples)
    assert examples[0].input == dummy_string_example.input
    assert examples[0].expected_output == dummy_string_example.expected_output


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_file_dataset_repository_returns_empty_list_for_not_existing_dataset_id(
    repository_fixture: str, request: FixtureRequest
) -> None:
    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)

    examples = dataset_repository.examples("some_name", DummyStringInput, type(None))

    assert examples == []


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_file_dataset_repository_can_delete_dataset(
    repository_fixture: str,
    dummy_string_examples: Iterable[Example[DummyStringInput, DummyStringOutput]],
    request: FixtureRequest,
) -> None:
    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)
    dataset_id = dataset_repository.create_dataset(dummy_string_examples)

    dataset_repository.delete_dataset(dataset_id)

    examples = dataset_repository.examples(
        dataset_id, DummyStringInput, DummyStringOutput
    )
    assert examples == []

    dataset_repository.delete_dataset(
        dataset_id
    )  # tests whether function is idempotent


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_file_dataset_repository_can_list_datasets(
    repository_fixture: str,
    dummy_string_example: Example[DummyStringInput, DummyStringOutput],
    request: FixtureRequest,
) -> None:
    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)
    examples = [dummy_string_example]
    dataset_id_1 = dataset_repository.create_dataset(examples=examples)
    dataset_id_2 = dataset_repository.create_dataset(examples=examples)
    dataset_names = dataset_repository.dataset_ids()
    assert sorted(dataset_names) == sorted([dataset_id_1, dataset_id_2])
