from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from intelligence_layer.connectors import DataClient, DataDataset, DatasetCreate
from intelligence_layer.evaluation.dataset.domain import (
    Dataset,
    Example,
)
from intelligence_layer.evaluation.dataset.studio_data_repository import (
    StudioDataRepository,
)


@pytest.fixture
def mock_data_client() -> Mock:
    return Mock(spec=DataClient)


@pytest.fixture
def studio_data_repository(mock_data_client: Mock) -> StudioDataRepository:
    return StudioDataRepository(repository_id="repo1", data_client=mock_data_client)


class InputExample(BaseModel):
    data: str


class ExpectedOutputExample(BaseModel):
    data: str


def test_create_dataset(
    studio_data_repository: StudioDataRepository, mock_data_client: Mock
) -> None:
    # Mock the data client's create_dataset method
    return_dataset_mock = Mock(spec=DataDataset)
    return_dataset_mock.dataset_id = "dataset1"
    return_dataset_mock.labels = ["label"]
    return_dataset_mock.metadata = {}
    return_dataset_mock.name = "Dataset 1"
    mock_data_client.create_dataset.return_value = return_dataset_mock

    # Create some example data

    examples = [
        Example(
            input=InputExample(data="input1"),
            expected_output=ExpectedOutputExample(data="output1"),
            id="example1",
        ),
        Example(
            input=InputExample(data="input2"),
            expected_output=ExpectedOutputExample(data="output2"),
            id="example2",
        ),
    ]

    # Call the method
    dataset = studio_data_repository.create_dataset(
        examples=examples, dataset_name="Dataset 1", labels={"label"}, metadata={}
    )

    # Assertions
    assert isinstance(dataset, Dataset)
    assert dataset.id == "dataset1"
    assert dataset.name == "Dataset 1"
    assert dataset.labels == {"label"}
    assert dataset.metadata == {}

    # Verify that the data client's create_dataset method was called with the correct parameters
    mock_data_client.create_dataset.assert_called_once_with(
        repository_id="repo1",
        dataset=DatasetCreate.model_validate(
            {
                "source_data": b'{"input":{"data":"input1"},"expected_output":{"data":"output1"},"id":"example1","metadata":null}\n{"input":{"data":"input2"},"expected_output":{"data":"output2"},"id":"example2","metadata":null}',
                "labels": ["label"],
                "name": "Dataset 1",
                "total_datapoints": 2,
                "metadata": {},
            }
        ),
    )


def test_delete_dataset(
    studio_data_repository: StudioDataRepository, mock_data_client: Mock
) -> None:
    # Call the method
    studio_data_repository.delete_dataset(dataset_id="dataset1")

    # Verify that the data client's delete_dataset method was called with the correct parameters
    mock_data_client.delete_dataset.assert_called_once_with(
        repository_id="repo1", dataset_id="dataset1"
    )


def test_dataset(
    studio_data_repository: StudioDataRepository, mock_data_client: Mock
) -> None:
    # Mock the data client's get_dataset method
    return_dataset_mock = Mock(spec=DataDataset)
    return_dataset_mock.dataset_id = "dataset1"
    return_dataset_mock.labels = []
    return_dataset_mock.metadata = {}
    return_dataset_mock.name = "Dataset 1"
    mock_data_client.get_dataset.return_value = return_dataset_mock
    # Call the method
    dataset = studio_data_repository.dataset(dataset_id="dataset1")

    # Assertions
    assert isinstance(dataset, Dataset)
    assert dataset.id == "dataset1"
    assert dataset.name == "Dataset 1"
    assert dataset.labels == set()
    assert dataset.metadata == {}

    # Verify that the data client's get_dataset method was called with the correct parameters
    mock_data_client.get_dataset.assert_called_once_with(
        repository_id="repo1", dataset_id="dataset1"
    )


def test_datasets(
    studio_data_repository: StudioDataRepository, mock_data_client: Mock
) -> None:
    # Mock the data client's list_datasets method
    return_dataset_mock = Mock(spec=DataDataset)
    return_dataset_mock.dataset_id = "dataset1"
    return_dataset_mock.labels = []
    return_dataset_mock.metadata = {}
    return_dataset_mock.name = "Dataset 1"

    return_dataset_mock_2 = Mock(spec=DataDataset)
    return_dataset_mock_2.dataset_id = "dataset2"
    return_dataset_mock_2.labels = []
    return_dataset_mock_2.metadata = {}
    return_dataset_mock_2.name = "Dataset 2"

    mock_data_client.list_datasets.return_value = [
        return_dataset_mock,
        return_dataset_mock_2,
    ]

    # Call the method
    datasets = list(studio_data_repository.datasets())

    # Assertions
    assert len(datasets) == 2
    assert isinstance(datasets[0], Dataset)
    assert datasets[0].id == "dataset1"
    assert datasets[0].name == "Dataset 1"
    assert datasets[0].labels == set()
    assert datasets[0].metadata == {}
    assert isinstance(datasets[1], Dataset)
    assert datasets[1].id == "dataset2"
    assert datasets[1].name == "Dataset 2"
    assert datasets[1].labels == set()
    assert datasets[1].metadata == {}

    # Verify that the data client's list_datasets method was called with the correct parameters
    mock_data_client.list_datasets.assert_called_once_with(repository_id="repo1")


def test_dataset_ids(
    studio_data_repository: StudioDataRepository, mock_data_client: Mock
) -> None:
    # Mock the data client's list_datasets method
    return_dataset_mock = Mock(spec=DataDataset)
    return_dataset_mock.dataset_id = "dataset1"
    return_dataset_mock.labels = ["label"]
    return_dataset_mock.metadata = {}
    return_dataset_mock.name = "Dataset 1"

    return_dataset_mock_2 = Mock(spec=DataDataset)
    return_dataset_mock_2.dataset_id = "dataset2"
    return_dataset_mock_2.labels = ["label"]
    return_dataset_mock_2.metadata = {}
    return_dataset_mock_2.name = "Dataset 2"

    mock_data_client.list_datasets.return_value = [
        return_dataset_mock,
        return_dataset_mock_2,
    ]

    # Call the method
    dataset_ids = list(studio_data_repository.dataset_ids())

    # Assertions
    assert len(dataset_ids) == 2
    assert dataset_ids[0] == "dataset1"
    assert dataset_ids[1] == "dataset2"

    # Verify that the data client's list_datasets method was called with the correct parameters
    mock_data_client.list_datasets.assert_called_once_with(repository_id="repo1")


def test_example(
    studio_data_repository: StudioDataRepository, mock_data_client: Mock
) -> None:
    # Mock the data client's stream_dataset method
    mock_data_client.stream_dataset.return_value = [
        b'{"input": {"data": "input1"}, "expected_output": {"data": "output1"}, "id": "example1"}',
        b'{"input": {"data": "input2"}, "expected_output": {"data": "output2"}, "id": "example2"}',
    ]

    # Call the method
    example = studio_data_repository.example(
        dataset_id="dataset1",
        example_id="example1",
        input_type=InputExample,
        expected_output_type=ExpectedOutputExample,
    )

    # Assertions
    assert isinstance(example, Example)
    assert example.input.data == "input1"
    assert example.expected_output.data == "output1"
    assert example.id == "example1"

    # Verify that the data client's stream_dataset method was called with the correct parameters
    mock_data_client.stream_dataset.assert_called_once_with(
        repository_id="repo1", dataset_id="dataset1"
    )


def test_examples(
    studio_data_repository: StudioDataRepository, mock_data_client: Mock
) -> None:
    # Mock the data client's stream_dataset method
    mock_data_client.stream_dataset.return_value = [
        b'{"input": {"data": "input1"}, "expected_output": {"data": "output1"}, "id": "example1"}',
        b'{"input": {"data": "input2"}, "expected_output": {"data": "output2"}, "id": "example2"}',
    ]

    # Call the method
    examples = list(
        studio_data_repository.examples(
            dataset_id="dataset1",
            input_type=InputExample,
            expected_output_type=ExpectedOutputExample,
        )
    )

    # Assertions
    assert len(examples) == 2
    assert isinstance(examples[0], Example)
    assert examples[0].input.data == "input1"
    assert examples[0].expected_output.data == "output1"
    assert examples[0].id == "example1"
    assert isinstance(examples[1], Example)
    assert examples[1].input.data == "input2"
    assert examples[1].expected_output.data == "output2"
    assert examples[1].id == "example2"

    # Verify that the data client's stream_dataset method was called with the correct parameters
    mock_data_client.stream_dataset.assert_called_once_with(
        repository_id="repo1", dataset_id="dataset1"
    )
