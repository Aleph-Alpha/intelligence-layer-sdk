from pytest import fixture
from intelligence_layer.core import InMemoryDatasetRepository, Example
from tests.conftest import DummyStringInput, DummyStringOutput


@fixture
def in_memory_dataset_repository() -> InMemoryDatasetRepository:
    return InMemoryDatasetRepository()


def test_in_memory_dataset_repository_can_store_dataset(
        in_memory_dataset_repository: InMemoryDatasetRepository,
        dummy_string_example: Example[DummyStringInput, DummyStringOutput]
    ) -> None:
    id = in_memory_dataset_repository.create_dataset(name="dummy dataset", examples=[(dummy_string_example.input, dummy_string_example.expected_output)])
    assert id in in_memory_dataset_repository._datasets
    dataset = in_memory_dataset_repository.dataset(id, input_type=DummyStringInput, expected_output_type=DummyStringOutput)
    assert dataset
    assert list(dataset.examples)[0].input == dummy_string_example.input
    assert list(dataset.examples)[0].expected_output == dummy_string_example.expected_output

def test_in_memory_dataset_repository_returns_none(
        in_memory_dataset_repository: InMemoryDatasetRepository,
) -> None:
    assert in_memory_dataset_repository.dataset("", DummyStringInput, None) is None

def test_in_memory_dataset_repository_can_delete_dataset(
        in_memory_dataset_repository: InMemoryDatasetRepository,
        dummy_string_example: Example[DummyStringInput, DummyStringOutput]
) -> None:
    id = in_memory_dataset_repository.create_dataset(name="dummy dataset", examples=[(dummy_string_example.input, dummy_string_example.expected_output)])
    assert in_memory_dataset_repository.dataset(id, DummyStringInput, DummyStringOutput)
    in_memory_dataset_repository.delete_dataset(id)
    assert in_memory_dataset_repository.dataset(id, DummyStringInput, DummyStringOutput) is None
    in_memory_dataset_repository.delete_dataset(id) #tests wether function is idempotent
    

def test_in_memory_dataset_repository_can_list_datasets(
        in_memory_dataset_repository: InMemoryDatasetRepository,
        dummy_string_example: Example[DummyStringInput, DummyStringOutput]
) -> None:
    dataset_name = "dummy_dataset"
    id0 = in_memory_dataset_repository.create_dataset(name=dataset_name, examples=[(dummy_string_example.input, dummy_string_example.expected_output)])
    id1 = in_memory_dataset_repository.create_dataset(name=dataset_name, examples=[(dummy_string_example.input, dummy_string_example.expected_output)])
    names_and_ids = in_memory_dataset_repository.list_datasets()
    assert names_and_ids[0] == (dataset_name,id0)
    assert names_and_ids[1] == (dataset_name,id1)