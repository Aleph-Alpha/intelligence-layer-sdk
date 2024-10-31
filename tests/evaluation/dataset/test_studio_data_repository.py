from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from intelligence_layer.connectors import DataClient, DataDataset
from intelligence_layer.connectors.studio.studio import StudioClient, StudioDataset
from intelligence_layer.evaluation.dataset.domain import (
    Dataset,
    Example,
)
from intelligence_layer.evaluation.dataset.studio_dataset_repository import (
    StudioDatasetRepository,
)


@pytest.fixture
def mock_data_client() -> Mock:
    return Mock(spec=DataClient)


@pytest.fixture
def mock_studio_client() -> Mock:
    return Mock(spec=StudioClient)


@pytest.fixture
def studio_dataset_repository(
    mock_data_client: Mock, mock_studio_client: Mock
) -> StudioDatasetRepository:
    return StudioDatasetRepository(
        repository_id="repo1",
        data_client=mock_data_client,
        studio_client=mock_studio_client,
    )


class InputExample(BaseModel):
    data: str


class ExpectedOutputExample(BaseModel):
    data: str


def test_create_dataset(studio_dataset_repository: StudioDatasetRepository) -> None:
    expected_dataset_id = "dataset1"
    studio_dataset_repository.studio_client.submit_dataset.return_value = (  # type: ignore
        expected_dataset_id
    )

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

    dataset = studio_dataset_repository.create_dataset(
        examples=examples,
        dataset_name="Dataset 1",
        labels={"label"},
        metadata={},
    )

    assert isinstance(dataset, Dataset)
    assert dataset.id == expected_dataset_id
    assert dataset.name == "Dataset 1"
    assert dataset.labels == {"label"}
    assert dataset.metadata == {}

    actual_call = studio_dataset_repository.studio_client.submit_dataset.call_args  # type: ignore
    submitted_dataset = actual_call[1]["dataset"]
    submitted_examples = actual_call[1]["examples"]

    expected_dataset = StudioDataset.model_validate(
        {
            "labels": ["label"],
            "name": "Dataset 1",
            "metadata": {},
        }
    )

    studio_examples = studio_dataset_repository.map_to_many_studio_example(examples)

    # Assertions
    assert submitted_dataset.labels == expected_dataset.labels
    assert submitted_dataset.name == expected_dataset.name
    assert submitted_dataset.metadata == expected_dataset.metadata
    assert submitted_examples == studio_examples


def test_delete_dataset(
    studio_dataset_repository: StudioDatasetRepository, mock_data_client: Mock
) -> None:
    studio_dataset_repository.delete_dataset(dataset_id="dataset1")

    mock_data_client.delete_dataset.assert_called_once_with(
        repository_id="repo1", dataset_id="dataset1"
    )


def test_dataset(
    studio_dataset_repository: StudioDatasetRepository, mock_data_client: Mock
) -> None:
    return_dataset_mock = Mock(spec=DataDataset)
    return_dataset_mock.dataset_id = "dataset1"
    return_dataset_mock.labels = []
    return_dataset_mock.metadata = {}
    return_dataset_mock.name = "Dataset 1"
    mock_data_client.get_dataset.return_value = return_dataset_mock
    dataset = studio_dataset_repository.dataset(dataset_id="dataset1")

    assert isinstance(dataset, Dataset)
    assert dataset.id == "dataset1"
    assert dataset.name == "Dataset 1"
    assert dataset.labels == set()
    assert dataset.metadata == {}

    mock_data_client.get_dataset.assert_called_once_with(
        repository_id="repo1", dataset_id="dataset1"
    )


def test_datasets(
    studio_dataset_repository: StudioDatasetRepository, mock_data_client: Mock
) -> None:
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

    datasets = list(studio_dataset_repository.datasets())

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

    mock_data_client.list_datasets.assert_called_once_with(repository_id="repo1")


def test_dataset_ids(
    studio_dataset_repository: StudioDatasetRepository, mock_data_client: Mock
) -> None:
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

    dataset_ids = list(studio_dataset_repository.dataset_ids())

    assert len(dataset_ids) == 2
    assert dataset_ids[0] == "dataset1"
    assert dataset_ids[1] == "dataset2"

    mock_data_client.list_datasets.assert_called_once_with(repository_id="repo1")


def test_example(
    studio_dataset_repository: StudioDatasetRepository, mock_data_client: Mock
) -> None:
    mock_data_client.stream_dataset.return_value = [
        b'{"input": {"data": "input1"}, "expected_output": {"data": "output1"}, "id": "example1"}',
        b'{"input": {"data": "input2"}, "expected_output": {"data": "output2"}, "id": "example2"}',
    ]

    example = studio_dataset_repository.example(
        dataset_id="dataset1",
        example_id="example1",
        input_type=InputExample,
        expected_output_type=ExpectedOutputExample,
    )

    assert isinstance(example, Example)
    assert example.input.data == "input1"
    assert example.expected_output.data == "output1"
    assert example.id == "example1"

    mock_data_client.stream_dataset.assert_called_once_with(
        repository_id="repo1", dataset_id="dataset1"
    )


def test_examples(
    studio_dataset_repository: StudioDatasetRepository, mock_data_client: Mock
) -> None:
    mock_data_client.stream_dataset.return_value = [
        b'{"input": {"data": "input1"}, "expected_output": {"data": "output1"}, "id": "example1"}',
        b'{"input": {"data": "input2"}, "expected_output": {"data": "output2"}, "id": "example2"}',
    ]

    examples = list(
        studio_dataset_repository.examples(
            dataset_id="dataset1",
            input_type=InputExample,
            expected_output_type=ExpectedOutputExample,
        )
    )

    assert len(examples) == 2
    assert isinstance(examples[0], Example)
    assert examples[0].input.data == "input1"
    assert examples[0].expected_output.data == "output1"
    assert examples[0].id == "example1"
    assert isinstance(examples[1], Example)
    assert examples[1].input.data == "input2"
    assert examples[1].expected_output.data == "output2"
    assert examples[1].id == "example2"

    mock_data_client.stream_dataset.assert_called_once_with(
        repository_id="repo1", dataset_id="dataset1"
    )
