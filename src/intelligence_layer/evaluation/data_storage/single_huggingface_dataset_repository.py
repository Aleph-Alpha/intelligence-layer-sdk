from typing import Generic, Iterable, Sequence

from pydantic import BaseModel
from intelligence_layer.core.task import Input
from intelligence_layer.evaluation.data_storage.dataset_repository import (
    DatasetRepository,
)
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset

from intelligence_layer.evaluation.domain import Dataset, Example, ExpectedOutput


class MultipleChoiceInput(BaseModel):
    question: str
    choices: Sequence[str]


class SingleHuggingfaceDatasetRepository(
    DatasetRepository, Generic[Input, ExpectedOutput]
):
    def __init__(
        self,
        huggingface_dataset: (
            DatasetDict | Dataset | IterableDatasetDict | IterableDataset
        ),
    ) -> None:
        self._huggingface_dataset = huggingface_dataset

    def create_dataset(
        self, examples: Iterable[Example[Input, ExpectedOutput]], dataset_name: str
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

    def examples(
        self,
        dataset_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
    ) -> Iterable[Example[Input, ExpectedOutput]]:

        answers = "ABCD"

        for index, sample in enumerate(self._huggingface_dataset["test"]):
            yield Example(
                input=MultipleChoiceInput(
                    question=sample["question"], choices=sample["choices"]
                ),
                expected_output=answers[sample["answer"]],
                id=str(index),
            )
