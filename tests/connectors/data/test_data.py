from collections.abc import Iterator
from typing import Any
from unittest.mock import Mock

import pytest
from requests import Session
from requests.exceptions import RequestException
from requests.models import Response

from intelligence_layer.connectors.data import (
    DataClient,
    DataDataset,
    DataInternalError,
    DataRepository,
    DataRepositoryCreate,
    DataResourceNotFound,
    DatasetCreate,
)


@pytest.fixture
def mock_session() -> Mock:
    return Mock(spec=Session)


@pytest.fixture
def data_client(mock_session: Mock) -> DataClient:
    return DataClient(token="some-token", session=mock_session)


def test_list_repositories(data_client: DataClient, mock_session: Mock) -> None:
    def return_json_override() -> dict[Any, Any]:
        return {
            "repositories": [
                {
                    "repositoryId": "repo1",
                    "name": "Repository 1",
                    "mutable": True,
                    "mediaType": "application/json",
                    "modality": "image",
                    "createdAt": "2022-01-01T00:00:00Z",
                    "updatedAt": "2022-01-01T00:00:00Z",
                },
                {
                    "repositoryId": "repo2",
                    "name": "Repository 2",
                    "mutable": False,
                    "mediaType": "application/csv",
                    "modality": "text",
                    "createdAt": "2022-01-01T00:00:00Z",
                    "updatedAt": "2022-01-01T00:00:00Z",
                },
            ]
        }

    mock_response = Mock(spec=Response)
    mock_response.status_code = 200
    mock_response.json.return_value = return_json_override()
    mock_session.request.return_value = mock_response

    # Call the method
    repositories = data_client.list_repositories()

    # Assertions
    assert len(repositories) == 2
    assert isinstance(repositories[0], DataRepository)
    assert repositories[0].repository_id == "repo1"
    assert repositories[0].name == "Repository 1"
    assert repositories[0].mutable is True
    assert repositories[0].media_type == "application/json"
    assert repositories[0].modality == "image"
    assert isinstance(repositories[1], DataRepository)
    assert repositories[1].repository_id == "repo2"
    assert repositories[1].name == "Repository 2"
    assert repositories[1].mutable is False
    assert repositories[1].media_type == "application/csv"
    assert repositories[1].modality == "text"

    # Verify the request was made with the correct parameters
    mock_session.request.assert_called_once_with(
        "GET",
        "http://localhost:8000/api/v1/repositories?page=0&size=20",
        headers={
            "Authorization": "Bearer some-token",
        },
    )


def test_list_repositories_handles_request_exception(
    data_client: DataClient, mock_session: Mock
) -> None:
    # Mock the request exception
    mock_session.request.side_effect = RequestException("Request failed")

    # Call the method
    with pytest.raises(DataInternalError):
        data_client.list_repositories()

    # Verify the request was made
    mock_session.request.assert_called_once()


def test_create_repository(data_client: DataClient, mock_session: Mock) -> None:
    # Mock the response

    def return_json_override() -> dict[Any, Any]:
        return {
            "repositoryId": "repo3",
            "name": "Repository 3",
            "mutable": True,
            "mediaType": "application/json",
            "modality": "text",
            "createdAt": "2022-01-01T00:00:00Z",
            "updatedAt": "2022-01-01T00:00:00Z",
        }

    mock_response = Mock(spec=Response)
    mock_response.status_code = 201
    mock_response.json.return_value = return_json_override()
    mock_session.request.return_value = mock_response

    # Call the method
    repository = data_client.create_repository(
        # Ignore because mypy does not support the dynamic transformation of pydantic.alias camelCase -> snake_case
        DataRepositoryCreate(
            name="Repository 3",
            mediaType="application/json",
            modality="text",  # type: ignore
        )
    )

    # Assertions
    assert isinstance(repository, DataRepository)
    assert repository.repository_id == "repo3"
    assert repository.name == "Repository 3"
    assert repository.mutable is True
    assert repository.media_type == "application/json"
    assert repository.modality == "text"

    # Verify the request was made with the correct parameters
    mock_session.request.assert_called_once_with(
        "POST",
        "http://localhost:8000/api/v1/repositories",
        headers={
            "Authorization": "Bearer some-token",
        },
        json={
            "name": "Repository 3",
            "mediaType": "application/json",
            "modality": "text",
        },
    )


def test_create_repository_handles_request_exception(
    data_client: DataClient, mock_session: Mock
) -> None:
    # Mock the request exception
    mock_session.request.side_effect = RequestException("Request failed")

    # Call the method
    with pytest.raises(DataInternalError):
        data_client.create_repository(
            # Ignore because mypy does not support the dynamic transformation of pydantic.alias camelCase -> snake_case
            DataRepositoryCreate(
                name="Repository 3",
                mediaType="application/json",
                modality="image",  # type: ignore
            )
        )

    # Verify the request was made
    mock_session.request.assert_called_once()


def test_get_repository(data_client: DataClient, mock_session: Mock) -> None:
    def return_json_override() -> dict[Any, Any]:
        return {
            "repositoryId": "repo3",
            "name": "Repository 3",
            "mutable": True,
            "mediaType": "application/json",
            "modality": "image",
            "createdAt": "2022-01-01T00:00:00Z",
            "updatedAt": "2022-01-01T00:00:00Z",
        }

    mock_response = Mock(spec=Response)
    mock_response.status_code = 200
    mock_response.json.return_value = return_json_override()
    mock_session.request.return_value = mock_response

    # Call the method
    repository = data_client.get_repository(repository_id="repo3")

    # Assertions
    assert isinstance(repository, DataRepository)
    assert repository.repository_id == "repo3"
    assert repository.name == "Repository 3"
    assert repository.mutable is True
    assert repository.media_type == "application/json"
    assert repository.modality == "image"

    # Verify the request was made with the correct parameters
    mock_session.request.assert_called_once_with(
        "GET",
        "http://localhost:8000/api/v1/repositories/repo3",
        headers={
            "Authorization": "Bearer some-token",
        },
    )


def test_get_repository_handles_request_exception(
    data_client: DataClient, mock_session: Mock
) -> None:
    # Mock the request exception
    mock_session.request.side_effect = RequestException("Request failed")

    # Call the method
    with pytest.raises(DataInternalError):
        data_client.get_repository(repository_id="repo3")

    # Verify the request was made
    mock_session.request.assert_called_once()


def test_create_dataset(data_client: DataClient, mock_session: Mock) -> None:
    def return_json_override() -> dict[Any, Any]:
        return {
            "repositoryId": "repo3",
            "datasetId": "dataset1",
            "labels": ["label1", "label2"],
            "totalDatapoints": 100,
            "createdAt": "2022-01-01T00:00:00Z",
            "updatedAt": "2022-01-01T00:00:00Z",
        }

    mock_response = Mock(spec=Response)
    mock_response.status_code = 201
    mock_response.json.return_value = return_json_override()
    mock_session.request.return_value = mock_response

    # Call the method
    dataset = data_client.create_dataset(
        repository_id="repo3",
        dataset=DatasetCreate(
            source_data=b"source_data",
            labels=["label1", "label2"],
            total_datapoints=100,
        ),
    )

    # Assertions
    assert isinstance(dataset, DataDataset)
    assert dataset.repository_id == "repo3"
    assert dataset.dataset_id == "dataset1"
    assert dataset.labels == ["label1", "label2"]
    assert dataset.total_datapoints == 100

    # Verify the request was made with the correct parameters
    mock_session.request.assert_called_once_with(
        "POST",
        "http://localhost:8000/api/v1/repositories/repo3/datasets",
        headers={
            "Authorization": "Bearer some-token",
        },
        files={
            "sourceData": b"source_data",
            "labels": "label1,label2",
            "totalDatapoints": 100,
        },
    )


def test_create_dataset_handles_request_exception(
    data_client: DataClient, mock_session: Mock
) -> None:
    # Mock the request exception
    mock_session.request.side_effect = RequestException("Request failed")

    # Call the method
    with pytest.raises(DataInternalError):
        data_client.create_dataset(
            repository_id="repo3",
            dataset=DatasetCreate(
                source_data=b"source_data",
                labels=["label1", "label2"],
                total_datapoints=100,
            ),
        )

    # Verify the request was made
    mock_session.request.assert_called_once()


def test_list_datasets(data_client: DataClient, mock_session: Mock) -> None:
    def return_json_override() -> dict[Any, Any]:
        return {
            "datasets": [
                {
                    "repositoryId": "repo3",
                    "datasetId": "dataset1",
                    "labels": ["label1", "label2"],
                    "totalDatapoints": 100,
                    "createdAt": "2022-01-01T00:00:00Z",
                    "updatedAt": "2022-01-01T00:00:00Z",
                },
                {
                    "repositoryId": "repo3",
                    "datasetId": "dataset2",
                    "labels": ["label3", "label4"],
                    "totalDatapoints": 200,
                    "createdAt": "2022-01-01T00:00:00Z",
                    "updatedAt": "2022-01-01T00:00:00Z",
                },
            ]
        }

    mock_response = Mock(spec=Response)
    mock_response.status_code = 200
    mock_response.json.return_value = return_json_override()
    mock_session.request.return_value = mock_response

    # Call the method
    datasets = data_client.list_datasets(repository_id="repo3")

    # Assertions
    assert len(datasets) == 2
    assert isinstance(datasets[0], DataDataset)
    assert datasets[0].repository_id == "repo3"
    assert datasets[0].dataset_id == "dataset1"
    assert datasets[0].labels == ["label1", "label2"]
    assert datasets[0].total_datapoints == 100
    assert isinstance(datasets[1], DataDataset)
    assert datasets[1].repository_id == "repo3"
    assert datasets[1].dataset_id == "dataset2"
    assert datasets[1].labels == ["label3", "label4"]
    assert datasets[1].total_datapoints == 200

    # Verify the request was made with the correct parameters
    mock_session.request.assert_called_once_with(
        "GET",
        "http://localhost:8000/api/v1/repositories/repo3/datasets?page=0&size=20",
        headers={
            "Authorization": "Bearer some-token",
        },
    )


def test_list_datasets_handles_request_exception(
    data_client: DataClient, mock_session: Mock
) -> None:
    # Mock the request exception
    mock_session.request.side_effect = RequestException("Request failed")

    # Call the method
    with pytest.raises(DataInternalError):
        data_client.list_datasets(repository_id="repo3")

    # Verify the request was made
    mock_session.request.assert_called_once()


def test_get_dataset(data_client: DataClient, mock_session: Mock) -> None:
    def return_json_override() -> dict[Any, Any]:
        return {
            "repositoryId": "repo3",
            "datasetId": "dataset1",
            "labels": ["label1", "label2"],
            "totalDatapoints": 100,
            "createdAt": "2022-01-01T00:00:00Z",
            "updatedAt": "2022-01-01T00:00:00Z",
        }

    mock_response = Mock(spec=Response)
    mock_response.status_code = 200
    mock_response.json.return_value = return_json_override()
    mock_session.request.return_value = mock_response

    # Call the method
    dataset = data_client.get_dataset(repository_id="repo3", dataset_id="dataset1")

    # Assertions
    assert isinstance(dataset, DataDataset)
    assert dataset.repository_id == "repo3"
    assert dataset.dataset_id == "dataset1"
    assert dataset.labels == ["label1", "label2"]
    assert dataset.total_datapoints == 100

    # Verify the request was made with the correct parameters
    mock_session.request.assert_called_once_with(
        "GET",
        "http://localhost:8000/api/v1/repositories/repo3/datasets/dataset1",
        headers={
            "Authorization": "Bearer some-token",
        },
    )


def test_get_dataset_handles_request_exception(
    data_client: DataClient, mock_session: Mock
) -> None:
    # Mock the request exception
    mock_session.request.side_effect = RequestException("Request failed")

    # Call the method
    with pytest.raises(DataInternalError):
        data_client.get_dataset(repository_id="repo3", dataset_id="dataset1")

    # Verify the request was made
    mock_session.request.assert_called_once()


def test_delete_dataset(data_client: DataClient, mock_session: Mock) -> None:
    # Call the method
    data_client.delete_dataset(repository_id="repo3", dataset_id="dataset1")

    # Verify the request was made with the correct parameters
    mock_session.request.assert_called_once_with(
        "DELETE",
        "http://localhost:8000/api/v1/repositories/repo3/datasets/dataset1",
        headers={
            "Authorization": "Bearer some-token",
        },
    )


def test_stream_dataset(data_client: DataClient, mock_session: Mock) -> None:
    expected_data = [b"data1", b"data2"]

    def mock_stream(*args: Any, **kwargs: Any) -> Iterator[Any]:
        yield from [b"data1", b"data2"]

    mock_response = Mock(spec=Response)
    mock_response.status_code = 200
    mock_response.iter_lines.return_value = mock_stream()
    mock_session.request.return_value = mock_response

    # Call the method
    stream = data_client.stream_dataset(repository_id="repo3", dataset_id="dataset1")

    # Assertions
    for expected, current in zip(expected_data, stream, strict=False):
        assert expected == current

    # Verify the request was made with the correct parameters
    mock_session.request.assert_called_once_with(
        "GET",
        "http://localhost:8000/api/v1/repositories/repo3/datasets/dataset1/datapoints",
        headers={
            "Authorization": "Bearer some-token",
        },
        stream=True,
    )


def test_do_request(data_client: DataClient, mock_session: Mock) -> None:
    # Mock the response
    mock_response = Response()
    mock_response.status_code = 200
    mock_session.request.return_value = mock_response

    # Call the method
    response = data_client._do_request("GET", "https://example.com")

    # Assertions
    assert response == mock_response

    # Verify the request was made with the correct parameters
    mock_session.request.assert_called_once_with(
        "GET",
        "https://example.com",
        headers={
            "Authorization": "Bearer some-token",
        },
    )


def test_do_request_handles_request_exception(
    data_client: DataClient, mock_session: Mock
) -> None:
    # Mock the request exception
    mock_session.request.side_effect = RequestException("Request failed")

    # Call the method
    with pytest.raises(DataInternalError):
        data_client._do_request("GET", "https://example.com")

    # Verify the request was made
    mock_session.request.assert_called_once()


def test_do_request_handles_status_code_exception(
    data_client: DataClient, mock_session: Mock
) -> None:
    # Mock the response
    mock_response = Response()
    mock_response.status_code = 404
    mock_session.request.return_value = mock_response

    # Call the method
    with pytest.raises(DataResourceNotFound):
        data_client._do_request("GET", "https://example.com")

    # Verify the request was made
    mock_session.request.assert_called_once()
