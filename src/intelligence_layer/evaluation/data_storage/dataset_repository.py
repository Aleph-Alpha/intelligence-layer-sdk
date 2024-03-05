from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, cast
from uuid import uuid4

from fsspec import AbstractFileSystem  # type: ignore
from fsspec.implementations.local import LocalFileSystem  # type: ignore

from intelligence_layer.core import Input, JsonSerializer, PydanticSerializable
from intelligence_layer.evaluation.domain import Example, ExpectedOutput


class DatasetRepository(ABC):
    """Base dataset repository interface.

    Provides methods to store and load datasets and their linked examples (:class:`Example`s).
    """

    @abstractmethod
    def create_dataset(
        self,
        examples: Iterable[Example[Input, ExpectedOutput]],
    ) -> str:
        """Creates a dataset from given :class:`Example`s and returns the ID of that dataset.

        Args:
            examples: An :class:`Iterable` of :class:`Example`s to be saved in the same dataset.

        Returns:
            The ID of the created dataset.
        """
        pass

    @abstractmethod
    def delete_dataset(self, dataset_id: str) -> None:
        """Deletes a dataset identified by the given dataset ID.

        Args:
            dataset_id: Dataset ID of the dataset to delete.
        """
        pass

    @abstractmethod
    def dataset_ids(self) -> Iterable[str]:
        """Returns all sorted dataset IDs.

        Returns:
            :class:`Iterable` of dataset IDs.
        """
        pass

    @abstractmethod
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

    @abstractmethod
    def examples(
        self,
        dataset_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
    ) -> Iterable[Example[Input, ExpectedOutput]]:
        """Returns all :class:`Example`s for the given dataset ID sorted by their ID.

        Args:
            dataset_id: Dataset ID whose examples should be retrieved.
            input_type: Input type of the example.
            expected_output_type: Expected output type of the example.

        Returns:
            :class:`Iterable` of :class`Example`s.
        """
        pass


class FileSystemDatasetRepository(DatasetRepository):
    _REPO_TYPE = "dataset"

    def __init__(self, fs: AbstractFileSystem, root_directory: str) -> None:
        super().__init__()
        assert root_directory[-1] != "/"
        self._fs = fs
        self._root_directory = root_directory

    def create_dataset(self, examples: Iterable[Example[Input, ExpectedOutput]]) -> str:
        dataset_id = str(uuid4())
        dataset_path = self._dataset_path(dataset_id)
        if self._fs.exists(dataset_path):
            raise ValueError(f"Dataset name {dataset_id} already taken")

        with self._fs.open(dataset_path, "w", encoding="utf-8") as examples_file:
            for example in examples:
                serialized_result = JsonSerializer(root=example)
                text = serialized_result.model_dump_json() + "\n"
                examples_file.write(text)
        return dataset_id

    def delete_dataset(self, dataset_id: str) -> None:
        dataset_path = self._dataset_path(dataset_id)
        try:
            self._fs.rm(dataset_path, recursive=True)
        except FileNotFoundError:
            pass

    def dataset_ids(self) -> Iterable[str]:
        return sorted(
            [
                Path(f["name"]).stem
                for f in self._fs.ls(self._root_directory, detail=True)
                if isinstance(f, Dict) and Path(f["name"]).suffix == ".jsonl"
            ]
        )

    def example(
        self,
        dataset_id: str,
        example_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
    ) -> Optional[Example[Input, ExpectedOutput]]:
        example_path = self._dataset_path(dataset_id)
        if not self._fs.exists(example_path):
            return None

        with self._fs.open(example_path, "r", encoding="utf-8") as examples_file:
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
        example_path = self._dataset_path(dataset_id)
        if not self._fs.exists(example_path):
            return []

        with self._fs.open(example_path, "r", encoding="utf-8") as examples_file:
            # Mypy does not accept dynamic types
            examples = [Example[input_type, expected_output_type].model_validate_json(json_data=example) for example in examples_file]  # type: ignore

        return sorted(examples, key=lambda example: example.id)

    def _dataset_path(self, dataset_id: str) -> str:
        return self._root_directory + f"/{dataset_id}.jsonl"


class InMemoryDatasetRepository(DatasetRepository):
    def __init__(self) -> None:
        self._datasets: dict[
            str, Sequence[Example[PydanticSerializable, PydanticSerializable]]
        ] = {}

    def create_dataset(
        self,
        examples: Iterable[Example[Input, ExpectedOutput]],
    ) -> str:
        dataset_id = str(uuid4())
        if dataset_id in self._datasets:
            raise ValueError(f"Dataset name {dataset_id} already taken")

        in_memory_examples = [
            cast(
                Example[PydanticSerializable, PydanticSerializable],
                example,
            )
            for example in examples
        ]
        self._datasets[dataset_id] = in_memory_examples
        return dataset_id

    def delete_dataset(self, dataset_id: str) -> None:
        self._datasets.pop(dataset_id, None)

    def dataset_ids(self) -> Iterable[str]:
        return sorted(list(self._datasets.keys()))

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
    ) -> Iterable[Example[Input, ExpectedOutput]]:
        return cast(
            Iterable[Example[Input, ExpectedOutput]],
            sorted(self._datasets.get(dataset_id, []), key=lambda example: example.id),
        )


class FileDatasetRepository(FileSystemDatasetRepository):
    def __init__(self, root_directory: Path) -> None:
        super().__init__(LocalFileSystem(), str(root_directory))
        root_directory.mkdir(parents=True, exist_ok=True)
