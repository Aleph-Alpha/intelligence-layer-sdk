from abc import ABC, abstractmethod
from functools import lru_cache
from os import getenv
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, cast
from uuid import uuid4

import wandb
from dotenv import load_dotenv
from fsspec import AbstractFileSystem  # type: ignore
from fsspec.implementations.local import LocalFileSystem  # type: ignore
from wandb.data_types import Table
from wandb.sdk.wandb_run import Run

from intelligence_layer.core import Input
from intelligence_layer.core.tracer import JsonSerializer, PydanticSerializable
from intelligence_layer.evaluation.domain import Example, ExpectedOutput


class DatasetRepository(ABC):
    @abstractmethod
    def create_dataset(
        self,
        examples: Iterable[Example[Input, ExpectedOutput]],
    ) -> str:
        pass

    @abstractmethod
    def examples_by_id(
        self,
        dataset_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
    ) -> Optional[Iterable[Example[Input, ExpectedOutput]]]:
        pass

    @abstractmethod
    def example(
        self,
        dataset_id: str,
        example_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
    ) -> Optional[Example[Input, ExpectedOutput]]:
        pass

    @abstractmethod
    def delete_dataset(self, dataset_id: str) -> None:
        pass

    @abstractmethod
    def list_datasets(self) -> Iterable[str]:
        pass


class WandbDatasetRepository(DatasetRepository):
    def __init__(self) -> None:
        self.team_name = "aleph-alpha-intelligence-layer-trial"
        self._run: Run | None = None

    def create_dataset(
        self,
        examples: Iterable[Example[Input, ExpectedOutput]],
    ) -> str:
        if self._run is None:
            raise ValueError("Run not started")
        dataset_id = str(uuid4())
        artifact = wandb.Artifact(name=dataset_id, type="dataset")
        table = Table(columns=["example"])  # type: ignore
        for example in examples:
            table.add_data(  # type: ignore
                example.model_dump_json(),
            )
        artifact.add(table, "dataset")
        self._run.log_artifact(artifact)
        return dataset_id

    def start_run(self, run: Run) -> None:
        self._run = run

    def finish_run(self) -> None:
        self._run = None

    def examples_by_id(
        self,
        dataset_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
    ) -> Optional[Iterable[Example[Input, ExpectedOutput]]]:
        table = self._get_dataset(dataset_id)
        return [Example[input_type, expected_output_type].model_validate_json(json_data=row[0]) for _, row in table.iterrows()]  # type: ignore

    @lru_cache(maxsize=1)
    def _get_dataset(self, id: str) -> Table:
        if self._run is None:
            raise ValueError("Run not started")
        artifact = self._run.use_artifact(
            f"{self.team_name}/{self._run.project_name()}/{id}:latest"
        )
        return artifact.get("dataset")  # type: ignore

    def example(
        self,
        dataset_id: str,
        example_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
    ) -> Optional[Example[Input, ExpectedOutput]]:
        examples = self.examples_by_id(dataset_id, input_type, expected_output_type)
        if examples is None:
            return None
        for example in examples:
            if example.id == example_id:
                return example
        return None

    def delete_dataset(self, dataset_id: str) -> None:
        raise NotImplementedError

    def list_datasets(self) -> Iterable[str]:
        raise NotImplementedError


class FileSystemDatasetRepository(DatasetRepository):
    _REPO_TYPE = "dataset"

    def __init__(self, fs: AbstractFileSystem, root_directory: str) -> None:
        super().__init__()
        assert root_directory[-1] != "/"
        self._fs = fs
        self._root_directory = root_directory

    def _dataset_path(self, dataset_id: str) -> str:
        return self._root_directory + f"/{dataset_id}.jsonl"

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
            # Mypy does not accept dynamic types
            for example in examples_file:
                validated_example = Example[input_type, expected_output_type].model_validate_json(json_data=example)  # type: ignore
                if validated_example.id == example_id:
                    return validated_example
        return None

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

    def examples_by_id(
        self,
        dataset_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
    ) -> Optional[Iterable[Example[Input, ExpectedOutput]]]:
        example_path = self._dataset_path(dataset_id)
        if not self._fs.exists(example_path):
            return None

        with self._fs.open(example_path, "r", encoding="utf-8") as examples_file:
            # Mypy does not accept dynamic types
            examples = [Example[input_type, expected_output_type].model_validate_json(json_data=example) for example in examples_file]  # type: ignore

        return (
            example
            for example in sorted(
                examples,
                key=lambda example: example.id if example else "",
            )
            if example
        )

    def delete_dataset(self, dataset_id: str) -> None:
        dataset_path = self._dataset_path(dataset_id)
        try:
            self._fs.rm(dataset_path, recursive=True)
        except FileNotFoundError:
            pass

    def list_datasets(self) -> Iterable[str]:
        return [
            Path(f["name"]).stem
            for f in self._fs.ls(self._root_directory, detail=True)
            if isinstance(f, Dict) and Path(f["name"]).suffix == ".jsonl"
        ]


class InMemoryDatasetRepository(DatasetRepository):
    """A repository to store datasets for evaluation."""

    def __init__(self) -> None:
        self._datasets: dict[
            str, Sequence[Example[PydanticSerializable, PydanticSerializable]]
        ] = {}

    def create_dataset(
        self,
        examples: Iterable[Example[Input, ExpectedOutput]],
    ) -> str:
        name = str(uuid4())
        if name in self._datasets:
            raise ValueError(f"Dataset name {name} already taken")
        in_memory_examples = [
            cast(
                Example[PydanticSerializable, PydanticSerializable],
                example,
            )
            for example in examples
        ]
        self._datasets[name] = in_memory_examples
        return name

    def examples_by_id(
        self,
        dataset_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
    ) -> Optional[Iterable[Example[Input, ExpectedOutput]]]:
        return cast(
            Optional[Iterable[Example[Input, ExpectedOutput]]],
            self._datasets.get(dataset_id),
        )

    def example(
        self,
        dataset_id: str,
        example_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
    ) -> Example[Input, ExpectedOutput] | None:
        examples = self.examples_by_id(dataset_id, input_type, expected_output_type)
        if examples is None:
            return None
        filtered = (e for e in examples if e.id == example_id)
        return next(filtered, None)

    def delete_dataset(self, dataset_id: str) -> None:
        self._datasets.pop(dataset_id, None)

    def list_datasets(self) -> Iterable[str]:
        return list(self._datasets.keys())


class FileDatasetRepository(FileSystemDatasetRepository):
    def __init__(self, root_directory: Path) -> None:
        super().__init__(LocalFileSystem(), str(root_directory))
        root_directory.mkdir(parents=True, exist_ok=True)
