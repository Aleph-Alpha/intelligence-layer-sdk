import os
import re
from typing import Any, Iterable, Mapping, Optional, Sequence

import requests
from pydantic import BaseModel, validator

from intelligence_layer.core.evaluation.domain import (
    AggregatedEvaluation,
    Dataset,
    Evaluation,
    Example,
    ExpectedOutput,
)
from intelligence_layer.core.evaluation.evaluator import (
    BaseEvaluator,
    EvaluationRepository,
)
from intelligence_layer.core.task import Input, Output, Task
from intelligence_layer.core.tracer import Tracer


class Field(BaseModel):
    name: str
    path: str

    @validator("path")
    def validate_path(cls, v: str) -> str:
        if not isinstance(v, str) or "." not in v:
            raise ValueError("Path must be a string with dot-separated values")
        return v

    def get_value(self, data_dict: dict) -> Any:
        keys = self.path.split(".")
        value = data_dict
        for key in keys:
            if key in value:
                value = value[key]
            else:
                raise KeyError(f"Key '{key}' not found in the provided dictionary.")
        return value


class ArgillaClient:
    def __init__(
        self, api_url: Optional[str] = None, api_key: Optional[str] = None
    ) -> None:
        self.api_url: str = api_url or os.environ["ARGILLA_API_URL"]
        self.api_key: str = api_key or os.environ["ARGILLA_API_KEY"]
        self.headers = {
            "accept": "application/json",
            "X-Argilla-Api-Key": self.api_key,
            "Content-Type": "application/json",
        }

    def upload(
        self,
        fields: Sequence[Field],
        examples: Sequence[tuple[Example[Input, ExpectedOutput], Output]],
    ) -> None:
        name = "name_test_dataset"
        workspace_id = "15657307-9780-44e5-87ba-4ad024a49a88"

        datasets = self._list_datasets(workspace_id)
        dataset_id = next(
            (item["id"] for item in datasets["items"] if item["name"] == name), None
        )
        if not dataset_id:
            dataset_id = self._create_dataset(name, workspace_id)["id"]
        field_names = [field.name for field in fields]

    def _list_datasets(self, workspace_id: str) -> Sequence[Any]:
        url = self.api_url + f"api/v1/me/datasets?workspace_id={workspace_id}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def _create_dataset(
        self, name: str, workspace_id: str, guidelines: str = "No guidelines."
    ) -> Mapping[str, Any]:
        url = self.api_url + "api/v1/datasets"
        data = {
            "name": name,
            "guidelines": guidelines,
            "workspace_id": workspace_id,
            "allow_extra_metadata": True,
        }
        response = requests.post(url, json=data, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def _list_fields(self, dataset_id: str) -> Sequence[Any]:
        url = self.api_url + f"api/v1/datasets/{dataset_id}/fields"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def _create_field(
        self, name: str, title: str, dataset_id: str
    ) -> Mapping[str, Any]:
        url = self.api_url + f"api/v1/datasets/{dataset_id}/fields"
        data = {
            "name": name,
            "title": title,
            "required": True,
            "settings": {"type": "text", "use_markdown": False},
        }
        response = requests.post(url, json=data, headers=self.headers)
        response.raise_for_status()
        return response.json()


class ArgillaEvaluator(
    BaseEvaluator[Input, Output, ExpectedOutput, Evaluation, AggregatedEvaluation]
):
    def __init__(
        self,
        task: Task[Input, Output],
        repository: EvaluationRepository,
        # mapping and feedback dataset must be provided here (I think), so it can be accessed in evaluate_example
        # or: have some kind of "ArgillaRepository" that saves the feedback dataset and have it accessed via this
    ) -> None:
        super().__init__(task, repository)

    def evaluate(
        self, example: Example[Input, ExpectedOutput], eval_id: str, output: Output
    ) -> None:
        # Inserts output into feedback dataset
        # according to mapping provided
        ...

    def aggregate(self, evaluations: Iterable[Evaluation]) -> AggregatedEvaluation:
        # don't worry about this now
        return None  # type: ignore

    def run_dataset_and_upload(
        dataset: Dataset[Input, ExpectedOutput],
        tracer: Optional[Tracer] = None,
        # providing feedback dataset & mapping here would be preferred (as it is stateful)
    ) -> str:  # should return eval_id
        return None  # type: ignore
