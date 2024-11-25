from collections.abc import Iterable
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from intelligence_layer.connectors import DataClient
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
def studio_dataset_repository(mock_studio_client: Mock) -> StudioDatasetRepository:
    return StudioDatasetRepository(
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

    dataset_name = "Dataset 1"
    dataset_labels = {"label"}

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
        dataset_name=dataset_name,
        labels=dataset_labels,
        metadata={},
    )

    assert isinstance(dataset, Dataset)
    assert dataset.id == expected_dataset_id
    assert dataset.name == dataset_name
    assert dataset.labels == dataset_labels
    assert dataset.metadata == {}

    studio_dataset_repository.studio_client.submit_dataset.assert_called_once()  # type: ignore

    actual_call = studio_dataset_repository.studio_client.submit_dataset.call_args  # type: ignore
    submitted_dataset = actual_call[1]["dataset"]
    submitted_examples = list(actual_call[1]["examples"])

    expected_dataset = StudioDataset(name=dataset_name, labels=dataset_labels)

    studio_examples = list(
        studio_dataset_repository.map_to_many_studio_example(examples)
    )

    # Assertions
    assert submitted_dataset.labels == expected_dataset.labels
    assert submitted_dataset.name == expected_dataset.name
    assert submitted_dataset.metadata == expected_dataset.metadata
    assert submitted_examples == studio_examples


def test_studio_client_is_only_called_once_when_examples_are_called(
    studio_dataset_repository: StudioDatasetRepository,
    mock_studio_client: StudioClient,
    sequence_examples: Iterable[Example[str, None]],
) -> None:
    dataset_id = "dataset_id"
    mock_studio_client.get_dataset_examples.return_value = sequence_examples  # type: ignore
    studio_dataset_repository.examples(dataset_id, str, type(None))
    studio_dataset_repository.examples(dataset_id, str, type(None))

    mock_studio_client.get_dataset_examples.assert_called_once()  # type: ignore
