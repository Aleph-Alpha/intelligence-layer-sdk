from pathlib import Path
from typing import Iterable, Optional

from fsspec import AbstractFileSystem  # type: ignore
from fsspec.implementations.local import LocalFileSystem  # type: ignore

from intelligence_layer.core import Input, JsonSerializer, PydanticSerializable
from intelligence_layer.evaluation.dataset.dataset_repository import DatasetRepository
from intelligence_layer.evaluation.dataset.domain import (
    Dataset,
    Example,
    ExpectedOutput,
)
from intelligence_layer.evaluation.infrastructure.file_system_based_repository import (
    FileSystemBasedRepository,
)


class FileSystemDatasetRepository(DatasetRepository, FileSystemBasedRepository):
    _REPO_TYPE = "dataset"

    def __init__(self, filesystem: AbstractFileSystem, root_directory: Path) -> None:
        assert str(root_directory)[-1] != "/"

        super().__init__(file_system=filesystem, root_directory=root_directory)

        self._dataset_root_directory().mkdir(parents=True, exist_ok=True)

    def create_dataset(
        self, examples: Iterable[Example[Input, ExpectedOutput]], dataset_name: str
    ) -> Dataset:
        dataset = Dataset(name=dataset_name)
        self._dataset_directory(dataset.id).mkdir(exist_ok=True)

        dataset_path = self._dataset_path(dataset.id)
        examples_path = self._dataset_examples_path(dataset.id)
        if self._file_system.exists(dataset_path) or self._file_system.exists(
            examples_path
        ):
            raise ValueError(
                f"One of the dataset files already exist for dataset {dataset}. This should not happen. Files: {dataset_path}, {examples_path}."
            )

        self._write_data(dataset_path, [dataset])
        self._write_data(examples_path, examples)

        return dataset

    def delete_dataset(self, dataset_id: str) -> None:
        try:
            self._file_system.rm(
                self.path_to_str(self._dataset_directory(dataset_id)), recursive=True
            )
        except FileNotFoundError:
            pass

    def dataset(self, dataset_id: str) -> Optional[Dataset]:
        file_path = self.path_to_str(self._dataset_path(dataset_id))
        if not self._file_system.exists(file_path):
            return None

        with self._file_system.open(file_path, "r", encoding="utf-8") as file_content:
            # we save only one dataset per file
            return [
                Dataset.model_validate_json(dataset_string)
                for dataset_string in file_content
            ][0]

    def dataset_ids(self) -> Iterable[str]:
        dataset_files = self._file_system.glob(
            path=self.path_to_str(self._dataset_root_directory()) + "/**/*.jsonl",
            maxdepth=2,
            detail=False,
        )
        return sorted([Path(f).stem for f in dataset_files])

    def example(
        self,
        dataset_id: str,
        example_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
    ) -> Optional[Example[Input, ExpectedOutput]]:
        example_path = self.path_to_str(self._dataset_examples_path(dataset_id))
        if not self._file_system.exists(example_path):
            return None

        with self._file_system.open(
            example_path, "r", encoding="utf-8"
        ) as examples_file:
            for example in examples_file:
                # mypy does not accept dynamic types
                validated_example = Example[input_type, expected_output_type].model_validate_json(json_data=example)  # type: ignore
                if validated_example.id == example_id:
                    return validated_example
        return None

    def examples(
        self,
        dataset_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
    ) -> Iterable[Example[Input, ExpectedOutput]]:
        example_path = self.path_to_str(self._dataset_examples_path(dataset_id))
        if not self._file_system.exists(example_path):
            return []

        with self._file_system.open(
            example_path, "r", encoding="utf-8"
        ) as examples_file:
            # Mypy does not accept dynamic types
            examples = [Example[input_type, expected_output_type].model_validate_json(json_data=example) for example in examples_file]  # type: ignore

        return sorted(examples, key=lambda example: example.id)

    def _dataset_root_directory(self) -> Path:
        return self._root_directory / "datasets"

    def _dataset_directory(self, dataset_id: str) -> Path:
        return self._dataset_root_directory() / f"{dataset_id}"

    def _dataset_path(self, dataset_id: str) -> Path:
        return self._dataset_directory(dataset_id) / f"{dataset_id}.json"

    def _dataset_examples_path(self, dataset_id: str) -> Path:
        return self._dataset_directory(dataset_id) / f"{dataset_id}.jsonl"

    def _write_data(
        self,
        file_path: Path,
        data_to_write: Iterable[PydanticSerializable],
    ) -> None:
        with self._file_system.open(
            self.path_to_str(file_path), "w", encoding="utf-8"
        ) as file:
            for data_chunk in data_to_write:
                serialized_result = JsonSerializer(root=data_chunk)
                json_string = serialized_result.model_dump_json() + "\n"
                file.write(json_string)


class FileDatasetRepository(FileSystemDatasetRepository):
    def __init__(self, root_directory: Path) -> None:
        super().__init__(LocalFileSystem(), root_directory)