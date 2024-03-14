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
        return super().create_dataset(examples, dataset_name)

    def dataset(self, dataset_id: str) -> Dataset | None:
        return super().dataset(dataset_id)

    def dataset_ids(self) -> Iterable[str]:
        return super().dataset_ids()

    def delete_dataset(self, dataset_id: str) -> None:
        return super().delete_dataset(dataset_id)

    def example(
        self,
        dataset_id: str,
        example_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
    ) -> Example[Input, ExpectedOutput] | None:
        return super().example(dataset_id, example_id, input_type, expected_output_type)

    def examples(
        self,
        dataset_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
    ) -> Iterable[Example[Input, ExpectedOutput]]:

        for question in self._huggingface_dataset["test"]["question"]:
            yield Example(
                input=MultipleChoiceInput(question=question, choices=[]),
                expected_output="",
            )
