from abc import ABC, abstractmethod
from typing import Iterable, Optional, Sequence, cast

from intelligence_layer.core.evaluation.domain import (
    Dataset,
    Example,
    ExpectedOutput,
    SequenceDataset,
)
from intelligence_layer.core.task import Input
from intelligence_layer.core.tracer import PydanticSerializable


class DatasetRepository(ABC):
    @abstractmethod
    def create_dataset(
        self,
        name: str,
        examples: Iterable[Example[Input, ExpectedOutput]],
    ) -> None:
        ...

    @abstractmethod
    def dataset(
        self,
        id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
    ) -> Optional[Dataset[Input, ExpectedOutput]]:
        ...

    @abstractmethod
    def delete_dataset(self, id: str) -> None:
        ...

    @abstractmethod
    def list_datasets(self) -> Sequence[str]:
        ...


class InMemoryDatasetRepository(DatasetRepository):
    """A repository to store datasets for evaluation."""

    def __init__(self) -> None:
        self._datasets: dict[
            str, Dataset[PydanticSerializable, PydanticSerializable]
        ] = {}

    def create_dataset(
        self,
        name: str,
        examples: Iterable[Example[Input, ExpectedOutput]],
    ) -> None:
        if name in self._datasets:
            raise ValueError("Dataset name already taken")
        in_memory_examples = [
            cast(
                Example[PydanticSerializable, PydanticSerializable],
                Example(input=example_input, expected_output=example_output),
            )
            for (example_input, example_output) in examples
        ]
        self._datasets[name] = SequenceDataset(name=name, examples=in_memory_examples)

    def dataset(
        self,
        id: str,
        input_type: type[Input],
        expected_output_type: Optional[type[ExpectedOutput]] = None,
    ) -> Optional[Dataset[Input, ExpectedOutput]]:
        return cast(Dataset[Input, ExpectedOutput], self._datasets.get(id))

    def delete_dataset(self, id: str) -> None:
        self._datasets.pop(id, None)

    def list_datasets(self) -> Sequence[str]:
        return list(self._datasets.keys())
