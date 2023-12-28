import json
from pathlib import Path
from typing import Dict, Iterable, Optional
from uuid import uuid4

import huggingface_hub  # type: ignore
from huggingface_hub import HfFileSystem, create_repo

from intelligence_layer.core.evaluation.domain import Example, ExpectedOutput
from intelligence_layer.core.evaluation.evaluator import DatasetRepository
from intelligence_layer.core.task import Input
from intelligence_layer.core.tracer import JsonSerializer


class HuggingFaceDatasetRepository(DatasetRepository):
    _REPO_TYPE = "dataset"

    def __init__(self, database_name: str, token: str, private: bool) -> None:
        super().__init__()
        assert database_name[-1] != "/"
        create_repo(
            database_name,
            token=token,
            repo_type=HuggingFaceDatasetRepository._REPO_TYPE,
            exist_ok=True,
            private=private,
        )
        self._database_name = database_name
        self._fs = HfFileSystem(token=token)
        self._root_directory = f"datasets/{database_name}"

    def delete_repository(self) -> None:
        huggingface_hub.delete_repo(
            database_name=self._database_name,
            token=self._fs.token,
            repo_type=HuggingFaceDatasetRepository._REPO_TYPE,
            missing_ok=True,
        )

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

        with self._fs.open(example_path, "r") as examples_file:
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

        with self._fs.open(dataset_path, "w") as examples_file:
            for example in examples:
                serialized_result = JsonSerializer(root=example)
                text = json.dumps(serialized_result.model_dump()) + "\n"
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

        with self._fs.open(example_path, "r") as examples_file:
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
