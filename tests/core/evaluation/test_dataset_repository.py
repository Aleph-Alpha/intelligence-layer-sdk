from pytest import raises

from intelligence_layer.core import Example, InMemoryDatasetRepository
from tests.conftest import DummyStringInput, DummyStringOutput


def test_in_memory_dataset_repository_can_store_dataset(
    dataset_repository: InMemoryDatasetRepository,
    dummy_string_example: Example[DummyStringInput, DummyStringOutput],
) -> None:
    dataset_name = "dummy dataset"
    dataset_repository.create_dataset(
        name=dataset_name, examples=[dummy_string_example]
    )
    assert dataset_name in dataset_repository._datasets
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


def test_in_memory_dataset_repository_throws_error_if_dataset_name_already_taken(
    dataset_repository: InMemoryDatasetRepository,
    string_dataset_name: str,
) -> None:
    with raises(ValueError) as e:
        dataset_repository.create_dataset(name=string_dataset_name, examples=[])
    assert "Dataset name already taken" in str(e)


def test_in_memory_dataset_repository_returns_none_for_nonexisting_dataset(
    dataset_repository: InMemoryDatasetRepository,
) -> None:
    assert dataset_repository.dataset("", DummyStringInput, None) is None


def test_in_memory_dataset_repository_can_delete_dataset(
    dataset_repository: InMemoryDatasetRepository, string_dataset_name: str
) -> None:
    dataset_repository.delete_dataset(string_dataset_name)
    assert (
        dataset_repository.dataset(
            string_dataset_name, DummyStringInput, DummyStringOutput
        )
        is None
    )
    dataset_repository.delete_dataset(
        string_dataset_name
    )  # tests whether function is idempotent


def test_in_memory_dataset_repository_can_list_datasets(
    dataset_repository: InMemoryDatasetRepository,
    dummy_string_example: Example[DummyStringInput, DummyStringOutput],
) -> None:
    dataset_name_1 = "dummy_dataset_1"
    dataset_name_2 = "dummy_dataset_2"
    examples = [dummy_string_example]
    dataset_repository.create_dataset(name=dataset_name_1, examples=examples)
    dataset_repository.create_dataset(name=dataset_name_2, examples=examples)
    dataset_names = dataset_repository.list_datasets()
    assert sorted(dataset_names) == sorted([dataset_name_1, dataset_name_2])
