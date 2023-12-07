from abc import ABC, abstractmethod
import os
from typing import Iterable, Mapping, Optional, Sequence, Union
from pydantic import BaseModel
import requests


class Field(BaseModel):
    name: str
    title: str


Record = Mapping[str, str]
ArgillaEvaluation = Mapping[str, Union[str, int, float, bool]]


class ArgillaClient(ABC):

    @abstractmethod
    def create_dataset(self, workspace_id: str, dataset_name: str, fields: Sequence[Field]) -> str:
        ...

    @abstractmethod
    def add_record(self, dataset_id: str, record: Record) -> None:
        ...

    @abstractmethod
    def evaluations(self, dataset_id: str) -> Iterable[ArgillaEvaluation]:
        ...


# class ArgillaClient:
#     def __init__(
#         self, api_url: Optional[str] = None, api_key: Optional[str] = None
#     ) -> None:
#         self.api_url: str = api_url or os.environ["ARGILLA_API_URL"]
#         self.api_key: str = api_key or os.environ["ARGILLA_API_KEY"]
#         self.headers = {
#             "accept": "application/json",
#             "X-Argilla-Api-Key": self.api_key,
#             "Content-Type": "application/json",
#         }

#     def upload(
#         self,
#         fields: Sequence[Field],
#         examples: Sequence[tuple[Example[Input, ExpectedOutput], Output]],
#     ) -> None:
#         name = "name_test_dataset"
#         workspace_id = "15657307-9780-44e5-87ba-4ad024a49a88"

#         datasets = self._list_datasets(workspace_id)
#         dataset_id = next(
#             (item["id"] for item in datasets["items"] if item["name"] == name), None
#         )
#         if not dataset_id:
#             dataset_id = self._create_dataset(name, workspace_id)["id"]
#         field_names = [field.name for field in fields]

#     def _list_datasets(self, workspace_id: str) -> Sequence[Any]:
#         url = self.api_url + f"api/v1/me/datasets?workspace_id={workspace_id}"
#         response = requests.get(url, headers=self.headers)
#         response.raise_for_status()
#         return response.json()

#     def _create_dataset(
#         self, name: str, workspace_id: str, guidelines: str = "No guidelines."
#     ) -> Mapping[str, Any]:
#         url = self.api_url + "api/v1/datasets"
#         data = {
#             "name": name,
#             "guidelines": guidelines,
#             "workspace_id": workspace_id,
#             "allow_extra_metadata": True,
#         }
#         response = requests.post(url, json=data, headers=self.headers)
#         response.raise_for_status()
#         return response.json()

#     def _list_fields(self, dataset_id: str) -> Sequence[Any]:
#         url = self.api_url + f"api/v1/datasets/{dataset_id}/fields"
#         response = requests.get(url, headers=self.headers)
#         response.raise_for_status()
#         return response.json()

#     def _create_field(
#         self, name: str, title: str, dataset_id: str
#     ) -> Mapping[str, Any]:
#         url = self.api_url + f"api/v1/datasets/{dataset_id}/fields"
#         data = {
#             "name": name,
#             "title": title,
#             "required": True,
#             "settings": {"type": "text", "use_markdown": False},
#         }
#         response = requests.post(url, json=data, headers=self.headers)
#         response.raise_for_status()
#         return response.json()
