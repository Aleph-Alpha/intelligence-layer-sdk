from pathlib import Path

from pytest import fixture, raises

from intelligence_layer.core import Example, FileDatasetRepository
from intelligence_layer.core.evaluation.domain import SequenceDataset
from tests.conftest import DummyStringInput, DummyStringOutput


@fixture
def file_dataset_repository(tmp_path: Path) -> FileDatasetRepository:
    return FileDatasetRepository(tmp_path)


def test_file_dataset_repository_can_store_dataset(
    file_dataset_repository: FileDatasetRepository,
    dummy_string_example: Example[DummyStringInput, DummyStringOutput],
) -> None:
    dataset_name = "dummy_dataset"
    file_dataset_repository.create_dataset(
        name=dataset_name, examples=[dummy_string_example]
    )
    dataset = file_dataset_repository.dataset(
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


def test_file_dataset_repository_throws_error_if_dataset_name_already_taken(
    file_dataset_repository: FileDatasetRepository,
    dummy_string_dataset: SequenceDataset[DummyStringInput, DummyStringOutput],
) -> None:
    file_dataset_repository.create_dataset(
        dummy_string_dataset.name, dummy_string_dataset.examples
    )
    with raises(ValueError) as e:
        file_dataset_repository.create_dataset(dummy_string_dataset.name, examples=[])
    assert dummy_string_dataset.name in str(e)


def test_file_dataset_repository_returns_none_for_nonexisting_dataset(
    file_dataset_repository: FileDatasetRepository,
) -> None:
    assert (
        file_dataset_repository.dataset("some_name", DummyStringInput, type(None))
        is None
    )


def test_file_dataset_repository_can_delete_dataset(
    file_dataset_repository: FileDatasetRepository,
    dummy_string_dataset: SequenceDataset[DummyStringInput, DummyStringOutput],
) -> None:
    file_dataset_repository.create_dataset(
        dummy_string_dataset.name, dummy_string_dataset.examples
    )
    file_dataset_repository.delete_dataset(dummy_string_dataset.name)
    assert (
        file_dataset_repository.dataset(
            dummy_string_dataset.name, DummyStringInput, DummyStringOutput
        )
        is None
    )
    file_dataset_repository.delete_dataset(
        dummy_string_dataset.name
    )  # tests whether function is idempotent


def test_file_dataset_repository_can_list_datasets(
    file_dataset_repository: FileDatasetRepository,
    dummy_string_example: Example[DummyStringInput, DummyStringOutput],
) -> None:
    dataset_name_1 = "dummy_dataset_1"
    dataset_name_2 = "dummy_dataset_2"
    examples = [dummy_string_example]
    file_dataset_repository.create_dataset(name=dataset_name_1, examples=examples)
    file_dataset_repository.create_dataset(name=dataset_name_2, examples=examples)
    dataset_names = file_dataset_repository.list_datasets()
    assert sorted(dataset_names) == sorted([dataset_name_1, dataset_name_2])
