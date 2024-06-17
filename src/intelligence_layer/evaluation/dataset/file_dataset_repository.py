import contextlib
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

from fsspec.implementations.local import LocalFileSystem  # type: ignore

from intelligence_layer.connectors.base.json_serializable import SerializableDict
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
    """A dataset repository that stores :class:`Dataset`s in files.

    It creates a single file per dataset and stores the :class:`Example`s as lines in this file.
    The format of the file is `.jsonl`.
    """

    _REPO_TYPE = "dataset"

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

        self.mkdir(self._dataset_directory(dataset.id))

        dataset_path = self._dataset_path(dataset.id)
        examples_path = self._dataset_examples_path(dataset.id)
        if self.exists(dataset_path) or self.exists(examples_path):
            raise ValueError(
                f"One of the dataset files already exist for dataset {dataset}. This should not happen. Files: {dataset_path}, {examples_path}."
            )

        self._write_data(dataset_path, [dataset])
        self._write_data(examples_path, examples)

        return dataset

    def delete_dataset(self, dataset_id: str) -> None:
        with contextlib.suppress(FileNotFoundError):
            self._file_system.rm(
                self.path_to_str(self._dataset_directory(dataset_id)), recursive=True
            )

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
        example_path = self._dataset_examples_path(dataset_id)
        if not self.exists(example_path.parent):
            raise ValueError(
                f"Repository does not contain a dataset with id: {dataset_id}"
            )
        if not self.exists(example_path):
            return None
        for example in self.examples(dataset_id, input_type, expected_output_type):
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
        example_path = self.path_to_str(self._dataset_examples_path(dataset_id))
        if not self._file_system.exists(example_path):
            raise ValueError(
                f"Repository does not contain a dataset with id: {dataset_id}"
            )

        with self._file_system.open(
            example_path, "r", encoding="utf-8"
        ) as examples_file:
            # Mypy does not accept dynamic types
            examples = []
            for example in examples_file:
                current_example = Example[
                    input_type, expected_output_type  # type: ignore
                ].model_validate_json(json_data=example)
                if current_example.id not in examples_to_skip:
                    examples.append(current_example)

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
        data = "\n".join(
            JsonSerializer(root=chunk).model_dump_json() for chunk in data_to_write
        )
        self.write_utf8(file_path, data, create_parents=True)


class FileDatasetRepository(FileSystemDatasetRepository):
    def __init__(self, root_directory: Path) -> None:
        super().__init__(LocalFileSystem(), root_directory)
