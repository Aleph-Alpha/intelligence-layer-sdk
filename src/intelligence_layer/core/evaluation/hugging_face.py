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

    # def create_dataset(self, examples: Iterable[Example[Input, ExpectedOutput]]) -> str:
    #    def gen():
    #        for example in examples:
    #            yield example
    #
    #    dataset_id = str(uuid4())
    #    # copy example when doing a run so example(db, id) is not needed
    #    # keep it simple, use Dataset and a single file
    #    # Remove example ids, use index into dataset
    #    # don't worry about runtime, they are small for now, remove example(id, ex_id)
    #    Dataset.from_generator(gen).push_to_hub(self._database_name, config_name=dataset_id, token=self._token)
    #    return dataset_id

    def delete_repository(self) -> None:
        huggingface_hub.delete_repo(
            database_name=self._database_name,
            token=self._fs.token,
            repo_type=HuggingFaceDatasetRepository._REPO_TYPE,
            missing_ok=True,
        )

    def _dataset_directory(self, dataset_id: str) -> str:
        return self._root_directory + "/" + dataset_id

    def _example_path(self, dataset_id: str, example_id: str) -> str:
        return self._dataset_directory(dataset_id) + "/" + example_id + ".json"

    def example(
        self,
        dataset_id: str,
        example_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
    ) -> Optional[Example[Input, ExpectedOutput]]:
        example_path = self._example_path(dataset_id, example_id)
        if not self._fs.exists(example_path):
            return None
        content = self._fs.read_text(example_path)
        # Mypy does not accept dynamic types
        return Example[input_type, expected_output_type].model_validate_json(json_data=content)  # type: ignore

    def create_dataset(self, examples: Iterable[Example[Input, ExpectedOutput]]) -> str:
        dataset_id = str(uuid4())
        dataset_dir = self._dataset_directory(dataset_id)
        if self._fs.exists(dataset_dir):
            raise ValueError(f"Dataset name {dataset_id} already taken")
        self._fs.mkdir(dataset_dir)
        for example in examples:
            serialized_result = JsonSerializer(root=example)
            example_path = self._example_path(dataset_id, example.id)
            text = serialized_result.model_dump_json(indent=2)
            self._fs.write_text(example_path, text)
        return dataset_id

    def examples_by_id(
        self,
        dataset_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
    ) -> Optional[Iterable[Example[Input, ExpectedOutput]]]:
        def load_example(
            path: str,
        ) -> Optional[Example[Input, ExpectedOutput]]:
            example_id = path[path.rfind("/") + 1 :]
            example_id = example_id[: example_id.rfind(".")]
            return self.example(
                dataset_id, example_id, input_type, expected_output_type
            )

        path = self._dataset_directory(dataset_id)
        if not self._fs.exists(path):
            return None

        example_files = self._fs.glob(path + "/*.json")
        files = list(load_example(file) for file in example_files)
        return (
            example
            for example in sorted(
                files,
                key=lambda example: example.id if example else "",
            )
            if example
        )

    def delete_dataset(self, dataset_id: str) -> None:
        dataset_path = self._dataset_directory(dataset_id)
        try:
            self._fs.rm(dataset_path, recursive=True)
        except FileNotFoundError:
            pass

    def list_datasets(self) -> Iterable[str]:
        return [
            Path(f["name"]).name
            for f in self._fs.ls(self._root_directory, detail=True)
            if isinstance(f, Dict) and f["type"] == "directory"
        ]
