from collections.abc import Iterable, Sequence
from typing import Optional, cast

from datasets import Dataset as HFDataset  # type: ignore
from datasets import DatasetDict, IterableDataset, IterableDatasetDict
from pydantic import BaseModel

from intelligence_layer.connectors.base.json_serializable import SerializableDict
from intelligence_layer.core.task import Input
from intelligence_layer.evaluation.dataset.dataset_repository import DatasetRepository
from intelligence_layer.evaluation.dataset.domain import (
    Dataset,
    Example,
    ExpectedOutput,
)


class MultipleChoiceInput(BaseModel):
    question: str
    choices: Sequence[str]


class SingleHuggingfaceDatasetRepository(DatasetRepository):
    def __init__(
        self,
        huggingface_dataset: (
            DatasetDict | HFDataset | IterableDatasetDict | IterableDataset
        ),
    ) -> None:
        self._huggingface_dataset = huggingface_dataset

    def create_dataset(
        self,
        examples: Iterable[Example[Input, ExpectedOutput]],
        dataset_name: str,
        id: str | None = None,
        labels: set[str] | None = None,
        metadata: SerializableDict | None = None,
    ) -> Dataset:
        raise NotImplementedError

    def dataset(self, dataset_id: str) -> Dataset | None:
        raise NotImplementedError

    def dataset_ids(self) -> Iterable[str]:
        raise NotImplementedError

    def delete_dataset(self, dataset_id: str) -> None:
        raise NotImplementedError

    def example(
        self,
        dataset_id: str,
        example_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
    ) -> Example[Input, ExpectedOutput] | None:
        examples = self.examples(
            dataset_id=dataset_id,
            input_type=input_type,
            expected_output_type=expected_output_type,
        )

        for example in examples:
            if example.id == example_id:
                return example
        return None

    def examples(
        self,
        dataset_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
        examples_to_skip: Optional[frozenset[str]] = None,
    ) -> Iterable[Example[Input, ExpectedOutput]]:
        examples_to_skip = examples_to_skip or frozenset()
        answers = "ABCD"
        assert input_type == MultipleChoiceInput
        assert expected_output_type is str
        for index, sample in enumerate(self._huggingface_dataset["test"]):
            if str(index) not in examples_to_skip:
                yield Example(
                    input=cast(
                        Input,
                        MultipleChoiceInput(
                            question=sample["question"], choices=sample["choices"]
                        ),
                    ),
                    expected_output=cast(ExpectedOutput, answers[sample["answer"]]),
                    id=str(index),
                )
