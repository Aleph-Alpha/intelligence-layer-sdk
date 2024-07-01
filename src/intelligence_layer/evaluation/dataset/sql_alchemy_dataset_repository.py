
from collections.abc import Iterable
from typing import Optional

from sqlalchemy import create_engine

from intelligence_layer.connectors.base.json_serializable import SerializableDict
from intelligence_layer.core.task import Input
from intelligence_layer.evaluation.dataset.dataset_repository import DatasetRepository
from intelligence_layer.evaluation.dataset.domain import (
    Dataset,
    Example,
    ExpectedOutput,
)


class SQLAlchemyDatasetRepository(DatasetRepository):
    def __init__(self, url: str) -> None:
        super().__init__()
        self.engine = create_engine(url=url)

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
            The created :class:`Dataset`.
        """
        pass

    def delete_dataset(self, dataset_id: str) -> None:
        """Deletes a dataset identified by the given dataset ID.

        Args:
            dataset_id: Dataset ID of the dataset to delete.
        """
        pass

    def dataset(self, dataset_id: str) -> Optional[Dataset]:
        """Returns a dataset identified by the given dataset ID.

        Args:
            dataset_id: Dataset ID of the dataset to delete.

        Returns:
            :class:`Dataset` if it was not, `None` otherwise.
        """
        pass

    def dataset_ids(self) -> Iterable[str]:
        """Returns all sorted dataset IDs.

        Returns:
            :class:`Iterable` of dataset IDs.
        """
        pass

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
        pass

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
        pass

