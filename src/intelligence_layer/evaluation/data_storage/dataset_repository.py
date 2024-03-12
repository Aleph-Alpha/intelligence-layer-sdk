from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, cast

from fsspec import AbstractFileSystem  # type: ignore
from fsspec.implementations.local import LocalFileSystem  # type: ignore

from intelligence_layer.core import Input, JsonSerializer, PydanticSerializable
from intelligence_layer.evaluation.domain import Dataset, Example, ExpectedOutput


class DatasetRepository(ABC):
    """Base dataset repository interface.

    Provides methods to store and load datasets and their linked examples (:class:`Example`s).
    """

    @abstractmethod
    def create_dataset(
        self, examples: Iterable[Example[Input, ExpectedOutput]], dataset_name: str
    ) -> Dataset:
        """Creates a dataset from given :class:`Example`s and returns the ID of that dataset.

        Args:
            examples: An :class:`Iterable` of :class:`Example`s to be saved in the same dataset.
            dataset_name: A name for the dataset.

        Returns:
            The created :class:`Dataset`.
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
    def dataset(self, dataset_id: str) -> Optional[Dataset]:
        """Returns a dataset identified by the given dataset ID.

        Args:
            dataset_id: Dataset ID of the dataset to delete.

        Returns:
            :class:`Dataset` if it was not, `None` otherwise.
        """
        pass

    def datasets(self) -> Iterable[Dataset]:
        """Returns all :class:`Dataset`s sorted by their ID.

        Returns:
            :class:`Sequence` of :class:`Dataset`s.
        """
        for dataset_id in self.dataset_ids():
            dataset = self.dataset(dataset_id)
            if dataset is not None:
                yield dataset

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

    def __init__(self, filesystem: AbstractFileSystem, root_directory: Path) -> None:
        super().__init__()

        assert str(root_directory)[-1] != "/"
        root_directory.mkdir(parents=True, exist_ok=True)

        self._file_system = filesystem
        self._root_directory = root_directory

    def create_dataset(
        self, examples: Iterable[Example[Input, ExpectedOutput]], dataset_name: str
    ) -> Dataset:
        dataset = Dataset(name=dataset_name)
        try:
            self._dataset_directory(dataset.id).mkdir(exist_ok=False)
        except OSError:
            raise ValueError(
                f"Created random dataset ID already exists for dataset {dataset}. This should not happen."
            )

        dataset_path = self._dataset_path(dataset.id)
        examples_path = self._dataset_examples_path(dataset.id)
        if self._file_system.exists(dataset_path) or self._file_system.exists(
            examples_path
        ):
            raise ValueError(
                f"One of the dataset files already exist for dataset {dataset}. This should not happen. Files: {dataset_path}, {examples_path}."
            )

        with self._file_system.open(
            str(dataset_path), "w", encoding="utf-8"
        ) as dataset_file:
            dataset_file.write(JsonSerializer(root=dataset).model_dump_json() + "\n")

        with self._file_system.open(
            str(examples_path), "w", encoding="utf-8"
        ) as examples_file:
            for example in examples:
                serialized_result = JsonSerializer(root=example)
                text = serialized_result.model_dump_json() + "\n"
                examples_file.write(text)

        return dataset

    def delete_dataset(self, dataset_id: str) -> None:
        try:
            self._file_system.rm(
                str(self._dataset_directory(dataset_id)), recursive=True
            )
        except FileNotFoundError:
            pass

    def dataset(self, dataset_id: str) -> Optional[Dataset]:
        file_path = self._dataset_path(dataset_id)
        if not file_path.exists():
            return None

        with self._file_system.open(
            str(file_path), "r", encoding="utf-8"
        ) as file_content:
            # we save only one dataset per file
            return [
                Dataset.model_validate_json(dataset_string)
                for dataset_string in file_content
            ][0]

    def dataset_ids(self) -> Iterable[str]:
        dataset_files = self._file_system.glob(
            path=str(self._root_directory) + "/**/*.json",
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
        if not self._file_system.exists(example_path):
            return None

        with self._file_system.open(
            str(example_path), "r", encoding="utf-8"
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
        example_path = self._dataset_examples_path(dataset_id)
        if not self._file_system.exists(example_path):
            return []

        with self._file_system.open(
            str(example_path), "r", encoding="utf-8"
        ) as examples_file:
            # Mypy does not accept dynamic types
            examples = [Example[input_type, expected_output_type].model_validate_json(json_data=example) for example in examples_file]  # type: ignore

        return sorted(examples, key=lambda example: example.id)

    def _dataset_directory(self, dataset_id: str) -> Path:
        return self._root_directory / f"{dataset_id}"

    def _dataset_path(self, dataset_id: str) -> Path:
        return self._dataset_directory(dataset_id) / f"{dataset_id}.json"

    def _dataset_examples_path(self, dataset_id: str) -> Path:
        return self._dataset_directory(dataset_id) / f"{dataset_id}.jsonl"


class InMemoryDatasetRepository(DatasetRepository):
    def __init__(self) -> None:
        self._datasets_and_examples: dict[
            str,
            Tuple[
                Dataset, Sequence[Example[PydanticSerializable, PydanticSerializable]]
            ],
        ] = {}

    def create_dataset(
        self, examples: Iterable[Example[Input, ExpectedOutput]], dataset_name: str
    ) -> Dataset:
        dataset = Dataset(name=dataset_name)
        if dataset.id in self._datasets_and_examples:
            raise ValueError(
                f"Created random dataset ID already exists for dataset {dataset}. This should not happen."
            )

        examples_casted = cast(
            Sequence[Example[PydanticSerializable, PydanticSerializable]], examples
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
        if dataset_id not in self._datasets_and_examples:
            return []
        return cast(
            Iterable[Example[Input, ExpectedOutput]],
            sorted(
                self._datasets_and_examples[dataset_id][1],
                key=lambda example: example.id,
            ),
        )


class FileDatasetRepository(FileSystemDatasetRepository):
    def __init__(self, root_directory: Path) -> None:
        super().__init__(LocalFileSystem(), root_directory)
