import io
import json
from collections.abc import Iterable
from typing import Optional

from intelligence_layer.connectors.base.json_serializable import (
    SerializableDict,
)
from intelligence_layer.connectors.data import DataClient
from intelligence_layer.connectors.data.models import DatasetCreate
from intelligence_layer.core import Input
from intelligence_layer.evaluation.dataset.dataset_repository import DatasetRepository
from intelligence_layer.evaluation.dataset.domain import (
    Dataset,
    Example,
    ExpectedOutput,
)


class StudioDataRepository(DatasetRepository):
    """Dataset repository interface with Data Platform."""

    def __init__(self, repository_id: str, data_client: DataClient) -> None:
        self.data_client = data_client
        self.repository_id = repository_id

    def create_dataset(
        self,
        examples: Iterable[Example[Input, ExpectedOutput]],
        dataset_name: str,
        id: str | None = None,
        labels: set[str] | None = None,
        metadata: SerializableDict | None = None,
    ) -> Dataset:
        """Creates a dataset from given :class:`Example`s and returns the ID of that dataset.

        Args:
            examples: An :class:`Iterable` of :class:`Example`s to be saved in the same dataset.
            dataset_name: A name for the dataset.
            id: The dataset ID. If `None`, an ID will be generated.
            labels: A list of labels for filtering. Defaults to an empty list.
            metadata: A dict for additional information about the dataset. Defaults to an empty dict.

        Returns:
           :class:`Dataset`
        """
        if id is not None:
            raise NotImplementedError(
                "Custom dataset IDs are not supported by the Data Platform"
            )

        source_data_list = [example.model_dump_json() for example in examples]
        remote_dataset = self.data_client.create_dataset(
            repository_id=self.repository_id,
            dataset=DatasetCreate(
                source_data=io.BytesIO("\n".join(source_data_list).encode()),
                labels=[label for label in labels] if labels is not None else [],
                total_units=len(source_data_list),
            ),
        )
        return Dataset(
            id=remote_dataset.dataset_id,
            name=dataset_name,  # Not implemented in data platform
            labels=set(remote_dataset.labels) if labels is not None else set(),
            metadata=metadata if metadata is not None else dict(),
        )

    def delete_dataset(self, dataset_id: str) -> None:
        """Deletes a dataset identified by the given dataset ID.

        Args:
            dataset_id: Dataset ID of the dataset to delete.
        """
        self.data_client.delete_dataset(
            repository_id=self.repository_id, dataset_id=dataset_id
        )

    def dataset(self, dataset_id: str) -> Optional[Dataset]:
        """Returns a dataset identified by the given dataset ID.

        Args:
            dataset_id: Dataset ID of the dataset to delete.

        Returns:
            :class:`Dataset` if it was not, `None` otherwise.
        """
        remote_dataset = self.data_client.get_dataset(
            repository_id=self.repository_id, dataset_id=dataset_id
        )
        return Dataset(
            id=remote_dataset.dataset_id,
            name="",  # Not implemented in data platform
            labels=set(remote_dataset.labels),
            metadata={},  # Not implemented in data platform
        )

    def datasets(self) -> Iterable[Dataset]:
        """Returns all :class:`Dataset`s sorted by their ID.

        Returns:
            :class:`Sequence` of :class:`Dataset`s.
        """
        for remote_dataset in self.data_client.list_datasets(
            repository_id=self.repository_id
        ):
            yield Dataset(
                id=remote_dataset.dataset_id,
                name="",  # Not implemented in data platform
                labels=set(remote_dataset.labels),
                metadata={},  # Not implemented in data platform
            )

    def dataset_ids(self) -> Iterable[str]:
        """Returns all sorted dataset IDs.

        Returns:
            :class:`Iterable` of dataset IDs.
        """
        datasets = self.data_client.list_datasets(repository_id=self.repository_id)
        return (dataset.dataset_id for dataset in datasets)

    def example(
        self,
        dataset_id: str,
        example_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
    ) -> Optional[Example[Input, ExpectedOutput]]:
        """Returns an :class:`Example` for the given dataset ID and example ID.

        Args:
            dataset_id: Dataset ID of the linked dataset.
            example_id: ID of the example to retrieve.
            input_type: Input type of the example.
            expected_output_type: Expected output type of the example.

        Returns:
            :class:`Example` if it was found, `None` otherwise.
        """
        stream = self.data_client.stream_dataset(
            repository_id=self.repository_id, dataset_id=dataset_id
        )
        for item in stream:
            data = json.loads(item.decode())
            if data["id"] == example_id:
                return Example[input_type, expected_output_type].model_validate_json(  # type: ignore
                    json_data=item
                )
        return None

    def examples(
        self,
        dataset_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
        examples_to_skip: Optional[frozenset[str]] = None,
    ) -> Iterable[Example[Input, ExpectedOutput]]:
        """Returns all :class:`Example`s for the given dataset ID sorted by their ID.

        Args:
            dataset_id: Dataset ID whose examples should be retrieved.
            input_type: Input type of the example.
            expected_output_type: Expected output type of the example.
            examples_to_skip: Optional list of example IDs. Those examples will be excluded from the output.

        Returns:
            :class:`Iterable` of :class`Example`s.
        """
        stream = self.data_client.stream_dataset(
            repository_id=self.repository_id, dataset_id=dataset_id
        )
        for item in stream:
            data = json.loads(item.decode())
            if examples_to_skip is not None and data["id"] in examples_to_skip:
                continue
            yield Example[input_type, expected_output_type].model_validate_json(  # type: ignore
                json_data=item
            )
