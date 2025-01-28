from collections.abc import Iterable, Sequence
from typing import Any, Optional, TypeVar, cast

from pydantic import TypeAdapter

from intelligence_layer.connectors.base.json_serializable import (
    SerializableDict,
)
from intelligence_layer.core import Input, PydanticSerializable
from intelligence_layer.evaluation.dataset.dataset_repository import DatasetRepository
from intelligence_layer.evaluation.dataset.domain import (
    Dataset,
    Example,
    ExpectedOutput,
)


class InMemoryDatasetRepository(DatasetRepository):
    def __init__(self) -> None:
        self._datasets_and_examples: dict[
            str,
            tuple[
                Dataset, Sequence[Example[PydanticSerializable, PydanticSerializable]]
            ],
        ] = {}

    def create_dataset(
        self,
        examples: Iterable[Example[Input, ExpectedOutput]],
        dataset_name: str,
        id: str | None = None,
        labels: set[str] | None = None,
        metadata: SerializableDict | None = None,
    ) -> Dataset:
        if metadata is None:
            metadata = dict()
        if labels is None:
            labels = set()
        dataset = Dataset(name=dataset_name, labels=labels, metadata=metadata)
        if id is not None:
            dataset.id = id
        if dataset.id in self._datasets_and_examples:
            if id:
                raise ValueError(
                    f"Cannot create dataset - dataset with given ID '{dataset.id}' already exists."
                )
            else:
                raise ValueError(
                    f"Newly assigned random dataset ID {dataset.id} already exists. This should not happen."
                )

        examples_casted = cast(
            Sequence[Example[PydanticSerializable, PydanticSerializable]],
            list(examples),
        )
        self._datasets_and_examples[dataset.id] = (dataset, examples_casted)

        return dataset

    def delete_dataset(self, dataset_id: str) -> None:
        self._datasets_and_examples.pop(dataset_id, None)

    def dataset(self, dataset_id: str) -> Optional[Dataset]:
        if dataset_id in self._datasets_and_examples:
            return self._datasets_and_examples[dataset_id][0]
        return None

    def dataset_ids(self) -> Iterable[str]:
        return sorted(list(self._datasets_and_examples.keys()))

    T = TypeVar("T")

    @staticmethod
    def _convert_to_type(data: Any, desired_type: T) -> T:
        if type(data) is desired_type:
            return data
        else:
            return TypeAdapter(desired_type).validate_python(data)

    def example(
        self,
        dataset_id: str,
        example_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
    ) -> Optional[Example[Input, ExpectedOutput]]:
        examples = self.examples(dataset_id, input_type, expected_output_type)
        filtered = (e for e in examples if e.id == example_id)
        return next(filtered, None)

    def examples(
        self,
        dataset_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
        examples_to_skip: Optional[frozenset[str]] = None,
    ) -> Iterable[Example[Input, ExpectedOutput]]:
        examples_to_skip = examples_to_skip or frozenset()
        if dataset_id not in self._datasets_and_examples:
            raise ValueError(
                f"Repository does not contain a dataset with id: {dataset_id}"
            )
        examples: list[Example[Input, ExpectedOutput]] = []
        for example in self._datasets_and_examples[dataset_id][1]:
            if example.id in examples_to_skip:
                continue
            converted_input = self._convert_to_type(example.input, input_type)
            converted_expected_output = self._convert_to_type(
                example.expected_output, expected_output_type
            )
            examples.append(
                Example[input_type, expected_output_type](  # type: ignore
                    id=example.id,
                    input=converted_input,
                    expected_output=converted_expected_output,
                    metadata=example.metadata,
                )
            )
        return sorted(
            examples,
            key=lambda example: example.id,
        )
