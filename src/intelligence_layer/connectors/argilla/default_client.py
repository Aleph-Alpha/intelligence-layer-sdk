import itertools
import os
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from http import HTTPStatus
from itertools import chain, count, islice
from typing import (
    Any,
    Optional,
    TypeVar,
    cast,
)
from uuid import uuid4

from pydantic import BaseModel, computed_field
from requests import HTTPError, Session
from requests.adapters import HTTPAdapter
from requests.structures import CaseInsensitiveDict
from urllib3 import Retry

from intelligence_layer.connectors.argilla.argilla_client import (
    ArgillaClient,
    ArgillaEvaluation,
    Record,
    RecordData,
)

T = TypeVar("T")


def batch_iterator(iterable: Iterable[T], batch_size: int) -> Iterable[list[T]]:
    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch


class Question(BaseModel):
    """Definition of an evaluation-question for an Argilla feedback dataset.

    Attributes:
        name: The name of the question. This is used to reference the questions in json-documents
        title: The title of the field. This is displayed in the Argilla UI to users that perform the manual evaluations.
        description: A more verbose description of the question.
            This is displayed in the Argilla UI to users that perform the manual evaluations.
        options: All integer options to answer this question

    """

    name: str
    title: str
    description: str
    options: Sequence[int]  # range: 1-10

    @computed_field  # type: ignore[misc]
    @property
    def settings(self) -> Mapping[str, Any]:
        return {
            "type": "rating",
            "options": [{"value": option} for option in self.options],
        }


class Field(BaseModel):
    """Definition of an Argilla feedback-dataset field.

    Attributes:
        name: The name of the field. This is used to reference the field in json-documents
        title: The title of the field. This is displayed in the Argilla UI to users that perform the manual evaluations.

    """

    name: str
    title: str


class DefaultArgillaClient(ArgillaClient):
    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        total_retries: int = 5,
    ) -> None:
        warnings.warn(
            "DefaultArgillaClient is deprecated. Use ArgillaClient instead.",
            DeprecationWarning,
        )

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

    def ensure_workspace_exists(self, workspace_name: str) -> str:
        """Retrieves the id of an argilla workspace with specified name or creates a new workspace if necessary.

        Args:
            workspace_name: the name of the workspace to be retrieved or created.

        Returns:
            The id of an argilla workspace with the given `workspace_name`.
        """
        try:
            return cast(str, self._create_workspace(workspace_name)["id"])
        except HTTPError as e:
            if e.response.status_code == HTTPStatus.CONFLICT:
                workspaces = self._list_workspaces()
                return next(
                    cast(str, item["id"])
                    for item in workspaces["items"]
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
            for field in fields:
                self._create_field(field.name, field.title, dataset_id)

            for question in questions:
                self._create_question(
                    question.name,
                    question.title,
                    question.description,
                    question.settings,
                    dataset_id,
                )
            self._publish_dataset(dataset_id)
            return dataset_id

        except HTTPError as e:
            if e.response.status_code == HTTPStatus.CONFLICT:
                raise ValueError(
                    f"Cannot create dataset with name '{dataset_name}', either the given dataset name, already exists"
                    f"or field name or question name are duplicates."
                ) from e
            raise e

    def ensure_dataset_exists(
        self,
        workspace_id: str,
        dataset_name: str,
        fields: Sequence[Field],
        questions: Sequence[Question],
    ) -> str:
        try:
            datasets = self._list_datasets(workspace_id)
            existing_dataset_id: str = next(
                cast(str, item["id"])
                for item in datasets["items"]
                if item["name"] == dataset_name
            )
            return existing_dataset_id
        except StopIteration:
            pass
        except HTTPError as e:
            raise e

        try:
            dataset_id: str = self.create_dataset(
                workspace_id, dataset_name, fields, questions
            )
        except HTTPError as e:
            raise e

        for field in fields:
            self._ignore_failure_status(
                frozenset([HTTPStatus.CONFLICT]),
                lambda field=field: self._create_field(
                    field.name, field.title, dataset_id
                ),
            )

        for question in questions:
            self._ignore_failure_status(
                frozenset([HTTPStatus.CONFLICT]),
                lambda question=question: self._create_question(
                    question.name,
                    question.title,
                    question.description,
                    question.settings,
                    dataset_id,
                ),
            )
        self._ignore_failure_status(
            frozenset([HTTPStatus.UNPROCESSABLE_ENTITY]),
            lambda: self._publish_dataset(dataset_id),
        )
        return dataset_id

    def _ignore_failure_status(
        self, expected_failure: frozenset[HTTPStatus], f: Callable[..., None]
    ) -> None:
        try:
            f()
        except HTTPError as e:
            if e.response.status_code not in expected_failure:
                raise e

    def add_record(self, dataset_id: str, record: RecordData) -> None:
        self._create_records([record], dataset_id)

    def add_records(self, dataset_id: str, records: Sequence[RecordData]) -> None:
        self._create_records(records, dataset_id)

    def evaluations(self, dataset_id: str) -> Iterable[ArgillaEvaluation]:
        def to_responses(
            json_responses: Sequence[Mapping[str, Any]],
        ) -> Mapping[str, int | float | bool | str]:
            return {
                question_name: json_response["value"]
                for json_response in json_responses
                for question_name, json_response in json_response["values"].items()
            }

        return (
            ArgillaEvaluation(
                example_id=json_evaluation["example_id"],
                record_id=json_evaluation["id"],
                responses=to_responses(json_evaluation["responses"]),
                metadata=json_evaluation["metadata"],
            )
            for json_evaluation in self._list_records(
                dataset_id,
                optional_params={
                    "response_status": "submitted",
                    "include": "responses",
                },
            )
        )

    def split_dataset(self, dataset_id: str, n_splits: int) -> None:
        self._create_metadata_property(dataset_id, n_splits)
        self._add_split_to_records(dataset_id, n_splits)

    def _create_metadata_property(self, dataset_id: str, n_splits: int) -> None:
        response = self.session.get(
            f"{self.api_url}api/v1/me/datasets/{dataset_id}/metadata-properties"
        )
        response.raise_for_status()
        existing_split_id = [
            property["id"]
            for property in response.json()["items"]
            if property["name"] == "split"
        ]
        if len(existing_split_id) > 0:
            self.session.delete(
                f"{self.api_url}api/v1/metadata-properties/{existing_split_id[0]}"
            )

        data = {
            "id": str(uuid4()),
            "name": "split",
            "title": "split",
            "settings": {"type": "terms", "values": [str(i) for i in range(n_splits)]},
            "visible_for_annotators": True,
        }

        response = self.session.post(
            f"{self.api_url}api/v1/datasets/{dataset_id}/metadata-properties",
            json=data,
        )
        response.raise_for_status()

    def _add_split_to_records(self, dataset_id: str, n_splits: int) -> None:
        records = self._list_records(dataset_id)
        splits = itertools.cycle(range(n_splits))
        records_and_splits = zip(records, splits, strict=False)

        def chunks(
            iterator: Iterable[tuple[Mapping[str, Any], int]], size: int
        ) -> Iterable[Iterable[tuple[Mapping[str, Any], int]]]:
            for first in iterator:
                yield chain([first], islice(iterator, size - 1))

        for chunk in chunks(
            records_and_splits, size=1000
        ):  # argilla has a limit of 1000 records per request
            data = {
                "items": [
                    {
                        "id": record["id"],
                        "metadata": {
                            **record["metadata"],
                            "example_id": record["example_id"],
                            "split": str(split),
                        },
                    }
                    for record, split in chunk
                ]
            }
            response = self.session.patch(
                f"{self.api_url}api/v1/datasets/{dataset_id}/records",
                json=data,
            )
            response.raise_for_status()

    def _evaluations(self, dataset_id: str) -> Mapping[str, Any]:
        response = self.session.get(
            self.api_url + f"api/v1/datasets/{dataset_id}/records",
            params={"response_status": "submitted", "include": "responses"},
        )
        response.raise_for_status()
        return cast(Mapping[str, Any], response.json())

    def records(self, dataset_id: str) -> Iterable[Record]:
        return (
            Record(
                id=json_record["id"],
                content=json_record["fields"],
                example_id=json_record["example_id"],
                metadata=json_record["metadata"],
            )
            for json_record in self._list_records(dataset_id)
        )

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

    def _list_workspaces(self) -> Mapping[str, Any]:
        url = self.api_url + "api/v1/me/workspaces"
        response = self.session.get(url)
        response.raise_for_status()
        return cast(Mapping[str, Any], response.json())

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
        settings: Mapping[str, Any],
        dataset_id: str,
    ) -> None:
        url = self.api_url + f"api/v1/datasets/{dataset_id}/questions"
        data = {
            "name": name,
            "title": title,
            "description": description,
            "required": True,
            "settings": settings,
        }
        response = self.session.post(url, json=data)
        response.raise_for_status()

    def _list_records(
        self,
        dataset_id: str,
        optional_params: Optional[Mapping[str, str]] = None,
        page_size: int = 50,
    ) -> Iterable[Mapping[str, Any]]:
        url = self.api_url + f"api/v1/datasets/{dataset_id}/records"
        for offset in count(0, page_size):
            params: Mapping[str, str | int] = {
                **(optional_params or {}),
                "offset": offset,
                "limit": page_size,
            }
            response = self.session.get(url, params=params)
            response.raise_for_status()
            records = response.json()["items"]
            if not records:
                break
            for record in records:
                metadata = record["metadata"]
                example_id = record["metadata"]["example_id"]
                del metadata["example_id"]
                record["example_id"] = example_id
            yield from cast(Sequence[Mapping[str, Any]], records)

    def _create_records(
        self,
        records: Sequence[RecordData],
        dataset_id: str,
    ) -> None:
        url = self.api_url + f"api/v1/datasets/{dataset_id}/records"
        for batch in batch_iterator(records, 200):
            data = {
                "items": [
                    {
                        "fields": record.content,
                        "metadata": {
                            **record.metadata,
                            "example_id": record.example_id,
                        },
                    }
                    for record in batch
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
