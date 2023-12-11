import os
from abc import ABC, abstractmethod
from http import HTTPStatus
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Union, cast

from pydantic import BaseModel
from pydantic import Field as PydanticField
from requests import HTTPError, Session
from requests.adapters import HTTPAdapter
from requests.structures import CaseInsensitiveDict
from urllib3 import Retry


class Field(BaseModel):
    name: str
    title: str


class Question(BaseModel):
    name: str
    title: str
    description: str
    options: Sequence[int]  # range: 1-10


class ArgillaEvaluation(BaseModel):
    record_id: str
    # maps question-names to response values
    responses: Mapping[str, Union[str, int, float, bool]]


class RecordData(BaseModel):
    content: Mapping[str, str]
    example_id: str
    metadata: Mapping[str, str] = PydanticField(default_factory=dict)


class Record(RecordData):
    id: str


class ArgillaClient(ABC):
    @abstractmethod
    def create_dataset(
        self,
        workspace_id: str,
        dataset_name: str,
        fields: Sequence[Field],
        questions: Sequence[Question],
    ) -> str:
        ...

    @abstractmethod
    def add_record(self, dataset_id: str, record: RecordData) -> None:
        ...

    @abstractmethod
    def evaluations(self, dataset_id: str) -> Iterable[ArgillaEvaluation]:
        ...


class DefaultArgillaClient(ArgillaClient):
    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        total_retries: int = 5,
    ) -> None:
        url = api_url or os.environ.get("ARGILLA_API_URL")
        key = api_key or os.environ.get("ARGILLA_API_KEY")
        if not (key and url):
            raise RuntimeError(
                "Environment variables ARGILLA_API_URL and ARGILLA_API_KEY must be defined to connect to an argilla instance"
            )
        self.api_url = url
        self.api_key = key
        retry_strategy = Retry(
            total=total_retries,
            backoff_factor=0.25,
            allowed_methods=["POST", "GET", "PUT"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = Session()
        self.session.headers = CaseInsensitiveDict({"X-Argilla-Api-Key": self.api_key})
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def create_workspace(self, workspace_name: str) -> str:
        try:
            return cast(str, self._create_workspace(workspace_name)["id"])
        except HTTPError as e:
            if e.response.status_code == HTTPStatus.CONFLICT:
                workspaces = self._list_workspaces()
                return next(
                    cast(str, item["id"])
                    for item in workspaces
                    if item["name"] == workspace_name
                )
            raise e

    def create_dataset(
        self,
        workspace_id: str,
        dataset_name: str,
        fields: Sequence[Field],
        questions: Sequence[Question],
    ) -> str:
        try:
            dataset_id: str = self._create_dataset(dataset_name, workspace_id)["id"]
        except HTTPError as e:
            if e.response.status_code == HTTPStatus.CONFLICT:
                datasets = self._list_datasets(workspace_id)
                dataset_id = next(
                    cast(str, item["id"])
                    for item in datasets["items"]
                    if item["name"] == dataset_name
                )
            else:
                raise e

        for field in fields:
            self._ignore_failure_status(
                frozenset([HTTPStatus.CONFLICT]),
                lambda: self._create_field(field.name, field.title, dataset_id),
            )

        for question in questions:
            self._ignore_failure_status(
                frozenset([HTTPStatus.CONFLICT]),
                lambda: self._create_question(
                    question.name,
                    question.title,
                    question.description,
                    question.options,
                    dataset_id,
                ),
            )
        self._ignore_failure_status(
            frozenset([HTTPStatus.UNPROCESSABLE_ENTITY]),
            lambda: self._publish_dataset(dataset_id),
        )
        return dataset_id

    def _ignore_failure_status(
        self, expected_failure: frozenset[HTTPStatus], f: Callable[[], None]
    ) -> None:
        try:
            f()
        except HTTPError as e:
            if e.response.status_code not in expected_failure:
                raise e

    def add_record(self, dataset_id: str, record: RecordData) -> None:
        try:
            self._create_record(
                record.content, record.metadata, record.example_id, dataset_id
            )
        except HTTPError as e:
            if e.response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR:
                records = self._list_records(dataset_id)
                if not any(
                    True
                    for item in records["items"]
                    if item["external_id"] == record.example_id
                ):
                    raise e

    def evaluations(self, dataset_id: str) -> Iterable[ArgillaEvaluation]:
        def to_responses(
            json_responses: Sequence[Mapping[str, Any]]
        ) -> Mapping[str, int | float | bool | str]:
            return {
                question_name: json_response["value"]
                for json_response in json_responses
                for question_name, json_response in json_response["values"].items()
            }

        return [
            ArgillaEvaluation(
                record_id=json_evaluation["id"],
                responses=to_responses(json_evaluation["responses"]),
            )
            for json_evaluation in self._evaluations(dataset_id)["items"]
        ]

    def _evaluations(self, dataset_id: str) -> Mapping[str, Any]:
        response = self.session.get(
            self.api_url + f"api/v1/datasets/{dataset_id}/records",
            params={"response_status": "submitted", "include": "responses"},
        )
        response.raise_for_status()
        return cast(Mapping[str, Any], response.json())

    def records(self, dataset_id: str) -> Iterable[Record]:
        json_records = self._list_records(dataset_id)
        return [
            Record(
                id=json_record["id"],
                content=json_record["fields"],
                example_id=json_record["external_id"],
                metadata=json_record["metadata"],
            )
            for json_record in json_records["items"]
        ]

    def create_evaluation(self, evaluation: ArgillaEvaluation) -> None:
        response = self.session.post(
            self.api_url + f"api/v1/records/{evaluation.record_id}/responses",
            json={
                "status": "submitted",
                "values": {
                    question_name: {"value": response_value}
                    for question_name, response_value in evaluation.responses.items()
                },
            },
        )
        response.raise_for_status()

    def _list_workspaces(self) -> Sequence[Any]:
        url = self.api_url + "api/workspaces"
        response = self.session.get(url)
        response.raise_for_status()
        return cast(Sequence[Any], response.json())

    def _create_workspace(self, workspace_name: str) -> Mapping[str, Any]:
        url = self.api_url + "api/workspaces"
        data = {
            "name": workspace_name,
        }
        response = self.session.post(url, json=data)
        response.raise_for_status()
        return cast(Mapping[str, Any], response.json())

    def _list_datasets(self, workspace_id: str) -> Mapping[str, Any]:
        url = self.api_url + f"api/v1/me/datasets?workspace_id={workspace_id}"
        response = self.session.get(url)
        response.raise_for_status()
        return cast(Mapping[str, Any], response.json())

    def _publish_dataset(self, dataset_id: str) -> None:
        url = self.api_url + f"api/v1/datasets/{dataset_id}/publish"
        response = self.session.put(url)
        response.raise_for_status()

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
        response = self.session.post(url, json=data)
        response.raise_for_status()
        return cast(Mapping[str, Any], response.json())

    def _list_fields(self, dataset_id: str) -> Sequence[Any]:
        url = self.api_url + f"api/v1/datasets/{dataset_id}/fields"
        response = self.session.get(url)
        response.raise_for_status()
        return cast(Sequence[Any], response.json())

    def _create_field(self, name: str, title: str, dataset_id: str) -> None:
        url = self.api_url + f"api/v1/datasets/{dataset_id}/fields"
        data = {
            "name": name,
            "title": title,
            "required": True,
            "settings": {"type": "text", "use_markdown": False},
        }
        response = self.session.post(url, json=data)
        response.raise_for_status()

    def _create_question(
        self,
        name: str,
        title: str,
        description: str,
        options: Sequence[int],
        dataset_id: str,
    ) -> None:
        url = self.api_url + f"api/v1/datasets/{dataset_id}/questions"
        data = {
            "name": name,
            "title": title,
            "description": description,
            "required": True,
            "settings": {
                "type": "rating",
                "options": [{"value": option} for option in options],
            },
        }
        response = self.session.post(url, json=data)
        response.raise_for_status()

    def _list_records(self, dataset_id: str) -> Mapping[str, Any]:
        url = self.api_url + f"api/v1/datasets/{dataset_id}/records"
        response = self.session.get(url)
        response.raise_for_status()
        return cast(Mapping[str, Any], response.json())

    def _create_record(
        self,
        content: Mapping[str, str],
        metadata: Mapping[str, str],
        example_id: str,
        dataset_id: str,
    ) -> None:
        url = self.api_url + f"api/v1/datasets/{dataset_id}/records"
        data = {
            "items": [
                {"fields": content, "metadata": metadata, "external_id": example_id}
            ]
        }
        response = self.session.post(url, json=data)
        response.raise_for_status()

    def delete_workspace(self, workspace_id: str) -> None:
        for dataset in self._list_datasets(workspace_id)["items"]:
            self._delete_dataset(dataset["id"])
        self._delete_workspace(workspace_id)

    def _delete_workspace(self, workspace_id: str) -> None:
        url = self.api_url + f"api/v1/workspaces/{workspace_id}"
        response = self.session.delete(url)
        response.raise_for_status()

    def _delete_dataset(self, dataset_id: str) -> None:
        url = self.api_url + f"api/v1/datasets/{dataset_id}"
        response = self.session.delete(url)
        response.raise_for_status()
