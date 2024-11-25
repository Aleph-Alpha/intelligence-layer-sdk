from collections.abc import Iterable, Sequence
from typing import Any
from uuid import UUID

from pytest import fixture

from intelligence_layer.connectors import StudioClient
from intelligence_layer.connectors.studio.studio import StudioDataset
from intelligence_layer.evaluation.dataset.domain import Example
from intelligence_layer.evaluation.dataset.in_memory_dataset_repository import (
    InMemoryDatasetRepository,
)
from intelligence_layer.evaluation.dataset.studio_dataset_repository import (
    StudioDatasetRepository,
)


@fixture
def labels() -> set[str]:
    return {"label1", "label2"}


@fixture
def metadata() -> dict[str, Any]:
    return {"key": "value"}


@fixture
def with_uploaded_dataset(
    studio_client: StudioClient, many_examples: Sequence[Example]
):
    dataset_repo = StudioDatasetRepository(studio_client)
    dataset = dataset_repo.create_dataset(many_examples, "my_dataset")

    return dataset


def test_can_upload_dataset_with_minimal_request_body(
    studio_client: StudioClient,
    examples: Sequence[Example[str, str]],
) -> None:
    dataset_repo = InMemoryDatasetRepository()
    dataset = dataset_repo.create_dataset(examples, "my_dataset")

    studio_dataset = StudioDatasetRepository.map_to_studio_dataset(dataset)
    studio_examples = StudioDatasetRepository.map_to_many_studio_example(examples)

    result = studio_client.submit_dataset(
        dataset=studio_dataset, examples=studio_examples
    )
    uuid = UUID(result)
    assert uuid


def test_can_upload_dataset_with_complete_request_body(
    studio_client: StudioClient,
    examples: Sequence[Example[str, str]],
    labels: set[str],
    metadata: dict[str, Any],
) -> None:
    dataset_repo = InMemoryDatasetRepository()
    dataset = dataset_repo.create_dataset(
        examples, "my_dataset", labels=labels, metadata=metadata
    )

    studio_dataset = StudioDatasetRepository.map_to_studio_dataset(dataset)
    studio_examples = StudioDatasetRepository.map_to_many_studio_example(examples)

    result = studio_client.submit_dataset(
        dataset=studio_dataset, examples=studio_examples
    )
    assert result


def test_get_many_dataset_examples(
    studio_client: StudioClient,
    many_examples: Iterable[Example[str, str]],
    with_uploaded_dataset: StudioDataset,
) -> None:
    received_examples = StudioDatasetRepository.map_to_many_example(
        studio_client.get_dataset_examples(
            with_uploaded_dataset.id, input_type=str, expected_output_type=str
        )
    )

    for received_example, given_example in zip(
        received_examples, many_examples, strict=True
    ):
        assert received_example.id == given_example.id
        assert received_example.input == given_example.input
        assert received_example.expected_output == given_example.expected_output
