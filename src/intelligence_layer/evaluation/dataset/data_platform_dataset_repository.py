import os
from typing import Iterable, Optional

import requests

from intelligence_layer.core import Input
from intelligence_layer.evaluation.dataset.dataset_repository import DatasetRepository
from intelligence_layer.evaluation.dataset.domain import (
    Dataset,
    Example,
    ExpectedOutput,
)


class DataPlatformDatasetRepository(DatasetRepository):
    DATA_STORAGE_URL = "https://data-storage.lengoo.com"
    DATA_PLATFORM_API_URL = "https://data.lengoo.com"

    def __init__(
        self, company_id: Optional[str] = None, storage_id: Optional[str] = None
    ) -> None:
        self.api_key = os.environ["DATAPLATFORM_API_KEY"]
        self._company_id = (
            company_id if company_id else os.environ["DATA_PLATFORM_COMPANY_ID"]
        )
        self._storage_id = (
            storage_id if storage_id else os.environ["DATA_PLATFORM_STORAGE_ID"]
        )
        self._dataset_type = {"dataset_type": "example_dataset"}
        self._auth_header = {"x-api-key": self.api_key}

    def create_dataset(
        self,
        examples: Iterable[Example[Input, ExpectedOutput]],
        dataset_name: str,
        id: str | None = None,
    ) -> Dataset:
        examples = list(examples)
        data = "\n".join([example.model_dump_json() for example in examples])
        byte_data = data.encode()
        response = requests.post(
            f"{self.DATA_STORAGE_URL}/api/v1/companies/{self._company_id}/storage/{self._storage_id}/files/",
            data=byte_data,
            headers={
                "File-Content-Type": "application/octet-stream",
            }
            | self._auth_header,
            stream=True,
        )
        response.raise_for_status()
        assert response.status_code == 201
        file_id = response.json()["file_id"]

        payload = {
            "source": file_id,
            "total_units": len(examples),
        } | self._dataset_type

        response = requests.post(
            f"{self.DATA_PLATFORM_API_URL}/v1.1/companies/{self._company_id}/datasets",
            json=payload,
            headers=self._auth_header,
        )
        response.raise_for_status()
        return Dataset(id=response.json()["dataset_id"], name=dataset_name)

    def delete_dataset(self, dataset_id: str) -> None:
        # TODO
        raise NotImplementedError()

    def dataset(self, dataset_id: str) -> Optional[Dataset]:
        response = requests.get(
            f"{self.DATA_PLATFORM_API_URL}/v1/companies/{self._company_id}/datasets/{dataset_id}?dataset_type=example_dataset",
            headers=self._auth_header,
        )
        response.raise_for_status()

        metadata = response.json()["metadata"]
        name = "unset"
        if metadata is not None and "name" in metadata:
            name = metadata["name"]
        return Dataset(id=dataset_id, name=name)

    def dataset_ids(self) -> Iterable[str]:
        # TODO paging
        response = requests.get(
            f"{self.DATA_PLATFORM_API_URL}/v1/companies/{self._company_id}/datasets?dataset_type=example_dataset&size=100",
            headers=self._auth_header,
        )
        list_of_datasets = response.json()["results"]
        for dataset in list_of_datasets:
            yield dataset["dataset_id"]

    def _get_download_url(self, dataset_id: str) -> str:
        response = requests.post(
            f"{self.DATA_PLATFORM_API_URL}/v1/companies/{self._company_id}/datasets",
            json={"file_id": [dataset_id]} | self._dataset_type,
            headers=self._auth_header,
        )
        if response.status_code == 404:
            raise ValueError(f"Dataset with id {dataset_id} does not exist")

        response.raise_for_status()
        return response.json()["presigned_url"]  # type: ignore

    def example(
        self,
        dataset_id: str,
        example_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
    ) -> Optional[Example[Input, ExpectedOutput]]:
        for example in self.examples(dataset_id, input_type, expected_output_type):
            if example.id == example_id:
                return example
        return None

    def examples(
        self,
        dataset_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
    ) -> Iterable[Example[Input, ExpectedOutput]]:
        # TODO
        download_url = self._get_download_url(dataset_id).replace(
            "data-storage.lengoo-internal.de", "data-storage.lengoo.com"
        )

        response = requests.get(download_url, stream=True, headers=self._auth_header)
        response.raise_for_status()

        # Mypy does not accept dynamic types
        examples = [
            Example[input_type, expected_output_type].model_validate_json(  # type: ignore
                json_data=example
            )
            for example in response.content.decode().split("\n")
        ]
        return sorted(examples, key=lambda example: example.id)
