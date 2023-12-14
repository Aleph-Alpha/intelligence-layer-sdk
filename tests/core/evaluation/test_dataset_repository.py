from pytest import fixture, raises

from intelligence_layer.core import Example, InMemoryDatasetRepository
from tests.conftest import DummyStringInput, DummyStringOutput


@fixture
def in_memory_dataset_repository() -> InMemoryDatasetRepository:
    return InMemoryDatasetRepository()


def test_in_memory_dataset_repository_can_store_dataset(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    dummy_string_example: Example[DummyStringInput, DummyStringOutput],
) -> None:
    dataset_name = "dummy dataset"
    in_memory_dataset_repository.create_dataset(
        name=dataset_name, examples=[dummy_string_example]
    )
    assert dataset_name in in_memory_dataset_repository._datasets
    dataset = in_memory_dataset_repository.dataset(
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
    in_memory_dataset_repository: InMemoryDatasetRepository,
    dataset_name: str,
) -> None:
    with raises(ValueError) as e:
        in_memory_dataset_repository.create_dataset(name=dataset_name, examples=[])
    assert "Dataset name already taken" in str(e)


def test_in_memory_dataset_repository_returns_none_for_nonexisting_dataset(
    in_memory_dataset_repository: InMemoryDatasetRepository,
) -> None:
    assert in_memory_dataset_repository.dataset("", DummyStringInput, None) is None


def test_in_memory_dataset_repository_can_delete_dataset(
    in_memory_dataset_repository: InMemoryDatasetRepository, dataset_name: str
) -> None:
    in_memory_dataset_repository.delete_dataset(dataset_name)
    assert (
        in_memory_dataset_repository.dataset(
            dataset_name, DummyStringInput, DummyStringOutput
        )
        is None
    )
    in_memory_dataset_repository.delete_dataset(
        dataset_name
    )  # tests whether function is idempotent


def test_in_memory_dataset_repository_can_list_datasets(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    dummy_string_example: Example[DummyStringInput, DummyStringOutput],
) -> None:
    dataset_name_1 = "dummy_dataset_1"
    dataset_name_2 = "dummy_dataset_2"
    examples = [dummy_string_example]
    in_memory_dataset_repository.create_dataset(name=dataset_name_1, examples=examples)
    in_memory_dataset_repository.create_dataset(name=dataset_name_2, examples=examples)
    dataset_names = in_memory_dataset_repository.list_datasets()
    assert sorted(dataset_names) == sorted([dataset_name_1, dataset_name_2])
