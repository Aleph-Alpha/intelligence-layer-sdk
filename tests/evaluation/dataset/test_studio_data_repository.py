from collections.abc import Iterable
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from intelligence_layer.connectors import DataClient
from intelligence_layer.connectors.studio.studio import (
    StudioClient,
    StudioDataset,
    StudioExample,
)
from intelligence_layer.evaluation.dataset.domain import (
    Dataset,
    Example,
)
from intelligence_layer.evaluation.dataset.studio_dataset_repository import (
    StudioDatasetRepository,
)
from tests.evaluation.conftest import DummyStringExpectedOutput, DummyStringInput


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


@pytest.fixture
def studio_examples() -> list[StudioExample[InputExample, ExpectedOutputExample]]:
    return [
        StudioExample(
            input=InputExample(data=f"input{i}"),
            expected_output=ExpectedOutputExample(data=f"output{i}"),
            id=f"example{i}",
        )
        for i in range(2)
    ]


def assertExamplesAreEqual(
    first_it: Iterable[Example | StudioExample],
    second_it: Iterable[Example | StudioExample],
):
    for first, second in zip(first_it, second_it, strict=True):
        assert first.id == second.id
        assert first.input == second.input
        assert first.expected_output == second.expected_output


def test_map_to_many_examples_works_properly(
    studio_examples: list[Example[InputExample, ExpectedOutputExample]],
) -> None:
    examples = StudioDatasetRepository.map_to_many_example(studio_examples)  # type: ignore

    assertExamplesAreEqual(studio_examples, examples)


def test_map_to_many_studio_examples_works_properly(
    studio_examples: list[StudioExample[InputExample, ExpectedOutputExample]],
) -> None:
    examples = StudioDatasetRepository.map_to_many_example(studio_examples)
    new_studio_examples = StudioDatasetRepository.map_to_many_studio_example(examples)
    assertExamplesAreEqual(studio_examples, new_studio_examples)


def test_create_dataset(
    studio_dataset_repository: StudioDatasetRepository,
    studio_examples: list[StudioExample[InputExample, ExpectedOutputExample]],
) -> None:
    expected_dataset_id = "dataset1"
    mock_submit_dataset: Mock = studio_dataset_repository.studio_client.submit_dataset  # type: ignore
    mock_submit_dataset.return_value = expected_dataset_id

    dataset_name = "Dataset 1"
    dataset_labels = {"label"}

    dataset = studio_dataset_repository.create_dataset(
        examples=StudioDatasetRepository.map_to_many_example(studio_examples),
        dataset_name=dataset_name,
        labels=dataset_labels,
        metadata={},
    )

    assert isinstance(dataset, Dataset)
    assert dataset.id == expected_dataset_id
    assert dataset.name == dataset_name
    assert dataset.labels == dataset_labels
    assert dataset.metadata == {}

    mock_submit_dataset.assert_called_once()

    actual_call = mock_submit_dataset.call_args
    submitted_dataset = actual_call[1]["dataset"]
    submitted_examples = list(actual_call[1]["examples"])

    expected_dataset = StudioDataset(name=dataset_name, labels=dataset_labels)

    # Assertions
    assert submitted_dataset.labels == expected_dataset.labels
    assert submitted_dataset.name == expected_dataset.name
    assert submitted_dataset.metadata == expected_dataset.metadata
    assert submitted_examples == studio_examples


def test_studio_client_is_only_called_once_when_examples_are_called(
    studio_dataset_repository: StudioDatasetRepository,
    mock_studio_client: StudioClient,
    sequence_examples: Iterable[Example[DummyStringInput, DummyStringExpectedOutput]],
) -> None:
    dataset_id = "dataset_id"
    mock_get_examples: Mock = mock_studio_client.get_dataset_examples  # type: ignore

    mock_get_examples.return_value = sequence_examples
    studio_dataset_repository.examples(
        dataset_id, DummyStringInput, DummyStringExpectedOutput
    )
    studio_dataset_repository.examples(
        dataset_id, DummyStringInput, DummyStringExpectedOutput
    )

    mock_get_examples.assert_called_once()


def test_examples_can_be_retrieved_properly(
    studio_dataset_repository: StudioDatasetRepository,
    mock_studio_client: StudioClient,
    studio_examples: list[StudioExample[InputExample, ExpectedOutputExample]],
) -> None:
    dataset_id = "dataset_id"
    mock_get_examples: Mock = mock_studio_client.get_dataset_examples  # type: ignore
    mock_get_examples.return_value = StudioDatasetRepository.map_to_many_example(
        studio_examples
    )
    examples = studio_dataset_repository.examples(
        dataset_id, InputExample, ExpectedOutputExample
    )
    assertExamplesAreEqual(studio_examples, examples)
    mock_get_examples.assert_called_once()
