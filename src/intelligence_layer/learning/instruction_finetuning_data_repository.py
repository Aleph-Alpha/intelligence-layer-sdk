from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Optional

from sqlalchemy import ColumnElement

from intelligence_layer.learning.models import (
    InstructionFinetuningSample,
)


class InstructionFinetuningDataRepository(ABC):
    @abstractmethod
    def store_sample(self, sample: InstructionFinetuningSample) -> str:
        """Stores a finetuning sample and returns its ID.

        Args:
            sample (InstructionFinetuningSample): The sample to store.

        Returns:
            str: The ID of the stored sample.
        """
        pass

    @abstractmethod
    def store_samples(
        self, samples: Iterable[InstructionFinetuningSample]
    ) -> list[str]:
        """Stores multiple finetuning samples and returns their IDs.

        Args:
            samples (Iterable[InstructionFinetuningSample]): The samples to store.

        Returns:
            list[str]: The IDs of the stored samples.
        """
        pass

    @abstractmethod
    def head(self, limit: Optional[int] = 100) -> Iterable[InstructionFinetuningSample]:
        """Returns the first `limit` samples.

        Args:
            limit: The number of samples to return. Defaults to 100.

        Returns:
            Iterable[InstructionFinetuningSample]: The first `limit` samples.
        """
        pass

    @abstractmethod
    def sample(self, id: str) -> Optional[InstructionFinetuningSample]:
        """Gets a finetuning sample by its ID.

        Args:
            id: The ID of the sample.

        Returns:
            The sample with the given ID, or None if not found.
        """
        pass

    @abstractmethod
    def samples(self, ids: Iterable[str]) -> Iterable[InstructionFinetuningSample]:
        """Gets multiple finetuning samples by their IDs.

        Args:
            ids: The IDs of the samples.

        Returns:
            The samples with the given IDs.
        """
        pass

    @abstractmethod
    def samples_with_filter(
        self, filter_expression: ColumnElement[bool], limit: Optional[int] = 100
    ) -> Iterable[InstructionFinetuningSample]:
        """Gets samples that match the given filter.

        Args:
            filter_expression: The filter expression.
            limit: The number of samples to return. Defaults to 100.

        Returns:
            The samples that match the filter.
        """
        pass

    @abstractmethod
    def delete_sample(self, id: str) -> None:
        """Deletes a finetuning sample by its ID.

        Args:
            id: The ID of the sample.
        """
        pass

    @abstractmethod
    def delete_samples(self, ids: Iterable[str]) -> None:
        """Deletes multiple finetuning samples by their IDs.

        Args:
            ids: The IDs of the samples.
        """
        pass
