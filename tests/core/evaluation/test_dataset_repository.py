from pathlib import Path

from pytest import FixtureRequest, fixture, mark, raises

from intelligence_layer.core import DatasetRepository, Example, FileDatasetRepository
from intelligence_layer.core.evaluation.domain import SequenceDataset
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
    dataset_name = "dummy_dataset"
    dataset_repository.create_dataset(
        name=dataset_name, examples=[dummy_string_example]
    )
    dataset = dataset_repository.dataset(
        dataset_name,
        input_type=DummyStringInput,
        expected_output_type=DummyStringOutput,
    )
    assert dataset
    assert list(dataset.examples)[0].input == dummy_string_example.input
    assert (
        list(dataset.examples)[0].expected_output
        == dummy_string_example.expected_output
    )


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_file_dataset_repository_throws_error_if_dataset_name_already_taken(
    repository_fixture: str,
    dummy_string_dataset: SequenceDataset[DummyStringInput, DummyStringOutput],
    request: FixtureRequest,
) -> None:
    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)
    dataset_repository.create_dataset(
        dummy_string_dataset.name, dummy_string_dataset.examples
    )
    with raises(ValueError) as e:
        dataset_repository.create_dataset(dummy_string_dataset.name, examples=[])
    assert dummy_string_dataset.name in str(e)


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_file_dataset_repository_returns_none_for_nonexisting_dataset(
    repository_fixture: str, request: FixtureRequest
) -> None:
    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)
    assert dataset_repository.dataset("some_name", DummyStringInput, type(None)) is None


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_file_dataset_repository_can_delete_dataset(
    repository_fixture: str,
    dummy_string_dataset: SequenceDataset[DummyStringInput, DummyStringOutput],
    request: FixtureRequest,
) -> None:
    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)
    dataset_repository.create_dataset(
        dummy_string_dataset.name, dummy_string_dataset.examples
    )
    dataset_repository.delete_dataset(dummy_string_dataset.name)
    assert (
        dataset_repository.dataset(
            dummy_string_dataset.name, DummyStringInput, DummyStringOutput
        )
        is None
    )
    dataset_repository.delete_dataset(
        dummy_string_dataset.name
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
    dataset_name_1 = "dummy_dataset_1"
    dataset_name_2 = "dummy_dataset_2"
    examples = [dummy_string_example]
    dataset_repository.create_dataset(name=dataset_name_1, examples=examples)
    dataset_repository.create_dataset(name=dataset_name_2, examples=examples)
    dataset_names = dataset_repository.list_datasets()
    assert sorted(dataset_names) == sorted([dataset_name_1, dataset_name_2])
