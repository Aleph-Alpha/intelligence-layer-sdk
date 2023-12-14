from pathlib import Path
from shutil import rmtree
from typing import Iterable, Optional, cast

from intelligence_layer.core.evaluation.domain import (
    Dataset,
    Example,
    ExpectedOutput,
    SequenceDataset,
)
from intelligence_layer.core.evaluation.evaluator import DatasetRepository
from intelligence_layer.core.task import Input
from intelligence_layer.core.tracer import JsonSerializer, PydanticSerializable


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
            raise ValueError(f"Dataset name {name} already taken")
        in_memory_examples = [
            cast(
                Example[PydanticSerializable, PydanticSerializable],
                example,
            )
            for example in examples
        ]
        self._datasets[name] = SequenceDataset(name=name, examples=in_memory_examples)

    def dataset(
        self,
        name: str,
        input_type: type[Input],
        expected_output_type: Optional[type[ExpectedOutput]] = None,
    ) -> Optional[Dataset[Input, ExpectedOutput]]:
        return cast(Dataset[Input, ExpectedOutput], self._datasets.get(name))

    def delete_dataset(self, id: str) -> None:
        self._datasets.pop(id, None)

    def list_datasets(self) -> Iterable[str]:
        return list(self._datasets.keys())


class FileDatasetRepository(DatasetRepository):
    def __init__(self, root_directory: Path) -> None:
        super().__init__()
        root_directory.mkdir(parents=True, exist_ok=True)
        self._root_directory = root_directory

    def _dataset_directory(self, dataset_name: str) -> Path:
        path = self._root_directory / dataset_name
        return path

    def _example_path(self, dataset_name: str, example_id: str) -> Path:
        return (self._dataset_directory(dataset_name) / example_id).with_suffix(".json")

    def _example(
        self,
        dataset_name: str,
        example_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
    ) -> Optional[Example[Input, ExpectedOutput]]:
        example_path = self._example_path(dataset_name, example_id)
        if not example_path.exists():
            return None
        content = example_path.read_text()
        # Mypy does not accept dynamic types
        return Example[input_type, expected_output_type].model_validate_json(json_data=content)  # type: ignore

    def create_dataset(
        self, name: str, examples: Iterable[Example[Input, ExpectedOutput]]
    ) -> None:
        dataset_dir = self._dataset_directory(name)
        if dataset_dir.exists():
            raise ValueError(f"Dataset name {name} already taken")
        dataset_dir.mkdir(exist_ok=True)
        for example in examples:
            serialized_result = JsonSerializer(root=example)
            self._example_path(name, example.id).write_text(
                serialized_result.model_dump_json(indent=2)
            )

    def dataset(
        self,
        name: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
    ) -> Dataset[Input, ExpectedOutput] | None:
        def load_example(
            path: Path,
        ) -> Optional[Example[Input, ExpectedOutput]]:
            id = path.with_suffix("").name
            return self._example(name, id, input_type, expected_output_type)

        path = self._dataset_directory(name)
        if not path.exists():
            return None

        example_files = path.glob("*.json")
        return SequenceDataset(
            name=name,
            examples=list(
                example
                for example in sorted(
                    (load_example(file) for file in example_files),
                    key=lambda example: example.id if example else "",
                )
                if example
            ),
        )

    def delete_dataset(self, name: str) -> None:
        dataset_path = self._dataset_directory(name)
        try:
            rmtree(dataset_path)
        except FileNotFoundError:
            pass

    def list_datasets(self) -> Iterable[str]:
        return (dataset_dir.name for dataset_dir in self._root_directory.iterdir())
