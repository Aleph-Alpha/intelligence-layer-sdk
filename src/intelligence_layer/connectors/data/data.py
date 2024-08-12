from collections.abc import Iterator
from http import HTTPStatus
from typing import Any, ClassVar
from urllib.parse import urlencode, urljoin

import requests
from requests import HTTPError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from intelligence_layer.connectors.data.exceptions import (
    DataInternalError,
    ExternalServiceUnavailable,
    ForbiddenError,
    InvalidInput,
    ResourceNotFound,
)
from intelligence_layer.connectors.data.models import (
    DataRepository,
    DataRepositoryCreate,
    Dataset,
    DatasetCreate,
)


class DataClient:
    """Data Client class to interact with the Data Platform API.

    Attributes:
    token: Token to authenticate with the Data Platform API
    base_data_platform_url: Base URL of the Data Platform API
    session: Session object to make requests to the Data Platform API


    Methods:
    list_repositories: List all the repositories
    create_repository: Create a new repository
    get_repository: Get a repository by ID
    create_dataset: Create a new dataset in a repository
    list_datasets: List all the datasets in a repository
    get_dataset: Get a dataset by ID
    delete_dataset: Delete a dataset by ID
    stream_dataset: Stream the data points of a dataset
    download_dataset: Download the data points of a dataset

    Examples:
    >>> client = DataClient(token="token")
    >>> repositories = client.list_repositories()
    >>> repository = client.create_repository(DataRepositoryCreate(name="name", mediaType="application/json", modality="text"))
    >>> dataset = client.create_dataset(repository_id=repository.repository_id, DatasetCreate(source_data=b"data", labels=["label"]))
    >>> datasets = client.list_datasets(repository_id=repository.repository_id)
    >>> dataset = client.get_dataset(repository_id=repository.repository_id, dataset_id=dataset.dataset_id)
    >>> client.delete_dataset(repository_id=repository.repository_id, dataset_id=dataset.dataset_id)
    >>> stream = client.stream_dataset(repository_id=repository.repository_id, dataset_id=dataset.dataset_id)
    """

    _status_code_to_exception: ClassVar = {
        HTTPStatus.SERVICE_UNAVAILABLE: ExternalServiceUnavailable,
        HTTPStatus.NOT_FOUND: ResourceNotFound,
        HTTPStatus.UNPROCESSABLE_ENTITY: InvalidInput,
        HTTPStatus.FORBIDDEN: ForbiddenError,
    }

    def __init__(
        self,
        token: str | None,
        base_data_platform_url: str = "http://localhost:8000",
        session: requests.Session | None = None,
    ) -> None:
        self.base_data_platform_url = base_data_platform_url
        self.headers = {
            **({"Authorization": f"Bearer {token}"} if token is not None else {}),
        }
        self.session = session or requests.Session()
        retry_strategy = Retry(
            total=3,  # Total number of retries
            backoff_factor=0.5,  # Exponential backoff factor
            status_forcelist=[
                429,
                500,
                502,
                503,
                504,
            ],  # Retry on these HTTP status codes
        )

        # Mount the retry strategy to the session
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def list_repositories(self, page: int = 0, size: int = 20) -> list[DataRepository]:
        """List all the repositories.

        Args:
            page: Page number
            size: Number of items per page

        Returns:
            List of DataRepository objects
        """
        url = urljoin(self.base_data_platform_url, "api/v1/repositories")
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
        DataRepository new object
        """
        url = urljoin(self.base_data_platform_url, "api/v1/repositories")
        response = self._do_request("POST", url, json=repository.dict())
        return DataRepository(**response.json())

    def get_repository(self, repository_id: str) -> DataRepository:
        """Get a repository by ID.

        Args:
            repository_id: Repository ID

        Returns:
        DataRepository object
        """
        url = urljoin(
            self.base_data_platform_url, f"api/v1/repositories/{repository_id}"
        )
        response = self._do_request("GET", url)
        return DataRepository(**response.json())

    def create_dataset(self, repository_id: str, dataset: DatasetCreate) -> Dataset:
        """Create a new dataset in a repository.

        Args:
            repository_id: Repository ID
            dataset: DatasetCreate object

        Returns:
            Dataset new object
        """
        url = urljoin(
            self.base_data_platform_url, f"api/v1/repositories/{repository_id}/datasets"
        )
        response = self._do_request(
            "POST",
            url,
            files={
                "source_data": dataset.source_data,
                "labels": ",".join(dataset.labels),
                "total_units": dataset.total_units,
            },
        )
        return Dataset(**response.json())

    def list_datasets(
        self, repository_id: str, page: int = 0, size: int = 20
    ) -> list[Dataset]:
        """List all the datasets in a repository.

        Args:
            repository_id: Repository ID
            page: Page number
            size: Number of items per page

        Returns:
        List of Dataset from a given repository
        """
        url = urljoin(
            self.base_data_platform_url, f"api/v1/repositories/{repository_id}/datasets"
        )
        query = urlencode({"page": page, "size": size})
        response = self._do_request("GET", f"{url}?{query}")
        datasets = response.json()
        return [Dataset(**dataset) for dataset in datasets["datasets"]]

    def get_dataset(self, repository_id: str, dataset_id: str) -> Dataset:
        """Get a dataset by ID.

        Args:
            repository_id: Repository ID
            dataset_id: Dataset ID

        Returns:
        Dataset new entity
        """
        url = urljoin(
            self.base_data_platform_url,
            f"api/v1/repositories/{repository_id}/datasets/{dataset_id}",
        )
        response = self._do_request("GET", url)
        return Dataset(**response.json())

    def delete_dataset(self, repository_id: str, dataset_id: str) -> None:
        """Delete a dataset by ID.

        Args:
            repository_id: Repository ID
            dataset_id: Dataset ID

        Returns:
        None
        """
        url = urljoin(
            self.base_data_platform_url,
            f"api/v1/repositories/{repository_id}/datasets/{dataset_id}",
        )
        self._do_request("DELETE", url)

    def stream_dataset(self, repository_id: str, dataset_id: str) -> Iterator[Any]:
        """Stream the data points of a dataset.

        Args:
            repository_id: Repository ID
            dataset_id: Dataset ID

        Returns:
        Iterator of data points
        """
        url = urljoin(
            self.base_data_platform_url,
            f"api/v1/repositories/{repository_id}/datasets/{dataset_id}/datapoints",
        )
        response = self._do_request("GET", url, stream=True)
        return response.iter_lines()

    def _do_request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        try:
            response = self.session.request(method, url, headers=self.headers, **kwargs)
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
