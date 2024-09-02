import json
import warnings
from collections.abc import Iterator
from http import HTTPStatus
from typing import Any, ClassVar
from urllib.parse import urlencode, urljoin

import requests
from requests import HTTPError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from intelligence_layer.connectors.data.exceptions import (
    DataError,
    DataExternalServiceUnavailable,
    DataForbiddenError,
    DataInternalError,
    DataInvalidInput,
    DataResourceNotFound,
)
from intelligence_layer.connectors.data.models import (
    DataDataset,
    DataRepository,
    DataRepositoryCreate,
    DatasetCreate,
)


class DataClient:
    """Client to interact with the Data Platform API.

    Attributes:
        headers: headers used in the request session
    """

    _status_code_to_exception: ClassVar[dict[int, type[DataError]]] = {
        HTTPStatus.SERVICE_UNAVAILABLE: DataExternalServiceUnavailable,
        HTTPStatus.NOT_FOUND: DataResourceNotFound,
        HTTPStatus.UNPROCESSABLE_ENTITY: DataInvalidInput,
        HTTPStatus.FORBIDDEN: DataForbiddenError,
    }

    def __init__(
        self,
        token: str | None,
        base_data_platform_url: str = "http://localhost:8000",
        session: requests.Session | None = None,
    ) -> None:
        """Initialize the Data Client.

        Args:
            token: Access token
            base_data_platform_url: Base URL of the Studio Data API. Defaults to "http://localhost:8000".
            session: a already created requests session. Defaults to None.
        """
        self._base_data_platform_url = base_data_platform_url
        self.headers = {
            **({"Authorization": f"Bearer {token}"} if token is not None else {}),
        }

        self._session = session or requests.Session()
        retry_strategy = Retry(
            total=3,  # Total number of retries
            backoff_factor=0.5,  # Exponential backoff factor
            status_forcelist=[
                HTTPStatus.TOO_MANY_REQUESTS,  # 429,
                HTTPStatus.INTERNAL_SERVER_ERROR,  # 500,
                HTTPStatus.BAD_GATEWAY,  # 502,
                HTTPStatus.SERVICE_UNAVAILABLE,  # 503,
                HTTPStatus.GATEWAY_TIMEOUT,  # 504,
            ],  # Retry on these HTTP status codes
        )

        # Mount the retry strategy to the session
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        warnings.warn("DataClient still in beta version and subject to change")

    def list_repositories(self, page: int = 0, size: int = 20) -> list[DataRepository]:
        """List all the repositories.

        Args:
            page: Page number. Defaults to 0
            size: Number of items per page. Defaults to 20

        Returns:
            List of :class:`DataRepository` objects
        """
        url = urljoin(self._base_data_platform_url, "api/v1/repositories")
        query = urlencode({"page": page, "size": size})
        response = self._do_request("GET", f"{url}?{query}")
        repositories = response.json()
        return [
            DataRepository(**repository) for repository in repositories["repositories"]
        ]

    def create_repository(self, repository: DataRepositoryCreate) -> DataRepository:
        """Create a new repository.

        Args:
            repository: DataRepositoryCreate object

        Returns:
            :class:`DataRepository` new object
        """
        url = urljoin(self._base_data_platform_url, "api/v1/repositories")
        response = self._do_request(
            "POST", url, json=repository.model_dump(by_alias=True)
        )
        return DataRepository(**response.json())

    def get_repository(self, repository_id: str) -> DataRepository:
        """Get a repository by ID.

        Args:
            repository_id: Repository ID

        Returns:
            :class:`DataRepository` object
        """
        url = urljoin(
            self._base_data_platform_url, f"api/v1/repositories/{repository_id}"
        )
        response = self._do_request("GET", url)
        return DataRepository(**response.json())

    def create_dataset(self, repository_id: str, dataset: DatasetCreate) -> DataDataset:
        """Create a new dataset in a repository.

        Args:
            repository_id: Repository ID
            dataset: :DatasetCreate object

        Returns:
            :class:`DataDataset` new object
        """
        url = urljoin(
            self._base_data_platform_url,
            f"api/v1/repositories/{repository_id}/datasets",
        )
        body = {
            "sourceData": dataset.source_data,
            "labels": ",".join(dataset.labels),
            "name": dataset.name,
            "totalDatapoints": dataset.total_datapoints,
            "metadata": json.dumps(dataset.metadata) if dataset.metadata else None,
        }
        response = self._do_request(
            "POST",
            url,
            files={k: v for k, v in body.items() if v not in [None, ""]},
        )
        return DataDataset(**response.json())

    def list_datasets(
        self, repository_id: str, page: int = 0, size: int = 20
    ) -> list[DataDataset]:
        """List all the datasets in a repository.

        Args:
            repository_id: Repository ID
            page: Page number. Defaults to 0
            size: Number of items per page. Defaults to 20

        Returns:
            List of :class:`DataDataset` from a given repository
        """
        url = urljoin(
            self._base_data_platform_url,
            f"api/v1/repositories/{repository_id}/datasets",
        )
        query = urlencode({"page": page, "size": size})
        response = self._do_request("GET", f"{url}?{query}")
        datasets = response.json()
        return [DataDataset(**dataset) for dataset in datasets["datasets"]]

    def get_dataset(self, repository_id: str, dataset_id: str) -> DataDataset:
        """Get a dataset by ID.

        Args:
            repository_id: Repository ID
            dataset_id: DataDataset ID

        Returns:
            :class:`DataDataset` object
        """
        url = urljoin(
            self._base_data_platform_url,
            f"api/v1/repositories/{repository_id}/datasets/{dataset_id}",
        )
        response = self._do_request("GET", url)
        return DataDataset(**response.json())

    def delete_dataset(self, repository_id: str, dataset_id: str) -> None:
        """Delete a dataset by ID.

        Args:
            repository_id: Repository ID
            dataset_id: DataDataset ID
        """
        url = urljoin(
            self._base_data_platform_url,
            f"api/v1/repositories/{repository_id}/datasets/{dataset_id}",
        )
        self._do_request("DELETE", url)

    def stream_dataset(self, repository_id: str, dataset_id: str) -> Iterator[Any]:
        """Stream the data points of a dataset.

        Args:
            repository_id: Repository ID
            dataset_id: DataDataset ID

        Returns:
            :class Iterator of datapoints(Any)
        """
        url = urljoin(
            self._base_data_platform_url,
            f"api/v1/repositories/{repository_id}/datasets/{dataset_id}/datapoints",
        )
        response = self._do_request("GET", url, stream=True)
        return response.iter_lines()

    def _do_request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        try:
            response = self._session.request(
                method, url, headers=self.headers, **kwargs
            )
            self._raise_for_status(response)
            return response
        except requests.RequestException as e:
            raise DataInternalError(str(e)) from e

    def _raise_for_status(self, response: requests.Response) -> None:
        try:
            response.raise_for_status()
        except HTTPError as e:
            exception_factory = self._status_code_to_exception.get(
                HTTPStatus(response.status_code), DataInternalError
            )
            raise exception_factory(
                response.text, HTTPStatus(response.status_code)
            ) from e
