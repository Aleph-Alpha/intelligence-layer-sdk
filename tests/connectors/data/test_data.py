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
    DataStage,
    DataStageCreate,
    DataFile,
    DataFileCreate,
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
                    "mediaType": "application/jsonlines",
                    "modality": "text",
                    "createdAt": "2022-01-01T00:00:00Z",
                    "updatedAt": "2022-01-01T00:00:00Z",
                },
                {
                    "repositoryId": "repo2",
                    "name": "Repository 2",
                    "mutable": False,
                    "mediaType": "application/jsonlines",
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

    repositories = data_client.list_repositories()

    assert len(repositories) == 2
    assert isinstance(repositories[0], DataRepository)
    assert repositories[0].repository_id == "repo1"
    assert repositories[0].name == "Repository 1"
    assert repositories[0].mutable is True
    assert repositories[0].media_type == "application/jsonlines"
    assert repositories[0].modality == "text"
    assert isinstance(repositories[1], DataRepository)
    assert repositories[1].repository_id == "repo2"
    assert repositories[1].name == "Repository 2"
    assert repositories[1].mutable is False
    assert repositories[1].media_type == "application/jsonlines"
    assert repositories[1].modality == "text"

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
    mock_session.request.side_effect = RequestException("Request failed")

    with pytest.raises(DataInternalError):
        data_client.list_repositories()

    mock_session.request.assert_called_once()


def test_create_repository(data_client: DataClient, mock_session: Mock) -> None:
    def return_json_override() -> dict[Any, Any]:
        return {
            "repositoryId": "repo3",
            "name": "Repository 3",
            "mutable": True,
            "mediaType": "application/jsonlines",
            "modality": "text",
            "createdAt": "2022-01-01T00:00:00Z",
            "updatedAt": "2022-01-01T00:00:00Z",
        }

    mock_response = Mock(spec=Response)
    mock_response.status_code = 201
    mock_response.json.return_value = return_json_override()
    mock_session.request.return_value = mock_response

    repository = data_client.create_repository(
        DataRepositoryCreate(
            name="Repository 3",
            mediaType="application/jsonlines",  # type: ignore
            modality="text",  # type: ignore
        )
    )

    assert isinstance(repository, DataRepository)
    assert repository.repository_id == "repo3"
    assert repository.name == "Repository 3"
    assert repository.mutable is True
    assert repository.media_type == "application/jsonlines"
    assert repository.modality == "text"

    mock_session.request.assert_called_once_with(
        "POST",
        "http://localhost:8000/api/v1/repositories",
        headers={
            "Authorization": "Bearer some-token",
        },
        json={
            "name": "Repository 3",
            "mediaType": "application/jsonlines",
            "modality": "text",
        },
    )


def test_create_repository_handles_request_exception(
    data_client: DataClient, mock_session: Mock
) -> None:
    mock_session.request.side_effect = RequestException("Request failed")

    with pytest.raises(DataInternalError):
        data_client.create_repository(
            DataRepositoryCreate(
                name="Repository 3",
                mediaType="application/jsonlines",  # type: ignore
                modality="text",  # type: ignore
            )
        )

    mock_session.request.assert_called_once()


def test_get_repository(data_client: DataClient, mock_session: Mock) -> None:
    def return_json_override() -> dict[Any, Any]:
        return {
            "repositoryId": "repo3",
            "name": "Repository 3",
            "mutable": True,
            "mediaType": "application/jsonlines",
            "modality": "text",
            "createdAt": "2022-01-01T00:00:00Z",
            "updatedAt": "2022-01-01T00:00:00Z",
        }

    mock_response = Mock(spec=Response)
    mock_response.status_code = 200
    mock_response.json.return_value = return_json_override()
    mock_session.request.return_value = mock_response

    repository = data_client.get_repository(repository_id="repo3")

    assert isinstance(repository, DataRepository)
    assert repository.repository_id == "repo3"
    assert repository.name == "Repository 3"
    assert repository.mutable is True
    assert repository.media_type == "application/jsonlines"
    assert repository.modality == "text"

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
    mock_session.request.side_effect = RequestException("Request failed")

    with pytest.raises(DataInternalError):
        data_client.get_repository(repository_id="repo3")

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

    dataset = data_client.create_dataset(
        repository_id="repo3",
        dataset=DatasetCreate(
            source_data=b"source_data",
            labels=["label1", "label2"],
            total_datapoints=100,
        ),
    )

    assert isinstance(dataset, DataDataset)
    assert dataset.repository_id == "repo3"
    assert dataset.dataset_id == "dataset1"
    assert dataset.labels == ["label1", "label2"]
    assert dataset.total_datapoints == 100

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
    mock_session.request.side_effect = RequestException("Request failed")

    with pytest.raises(DataInternalError):
        data_client.create_dataset(
            repository_id="repo3",
            dataset=DatasetCreate(
                source_data=b"source_data",
                labels=["label1", "label2"],
                total_datapoints=100,
            ),
        )

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

    datasets = data_client.list_datasets(repository_id="repo3")

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
    mock_session.request.side_effect = RequestException("Request failed")

    with pytest.raises(DataInternalError):
        data_client.list_datasets(repository_id="repo3")

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

    dataset = data_client.get_dataset(repository_id="repo3", dataset_id="dataset1")

    assert isinstance(dataset, DataDataset)
    assert dataset.repository_id == "repo3"
    assert dataset.dataset_id == "dataset1"
    assert dataset.labels == ["label1", "label2"]
    assert dataset.total_datapoints == 100

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
    mock_session.request.side_effect = RequestException("Request failed")

    with pytest.raises(DataInternalError):
        data_client.get_dataset(repository_id="repo3", dataset_id="dataset1")

    mock_session.request.assert_called_once()


def test_delete_dataset(data_client: DataClient, mock_session: Mock) -> None:
    data_client.delete_dataset(repository_id="repo3", dataset_id="dataset1")

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

    stream = data_client.stream_dataset(repository_id="repo3", dataset_id="dataset1")

    for expected, current in zip(expected_data, stream, strict=False):
        assert expected == current

    mock_session.request.assert_called_once_with(
        "GET",
        "http://localhost:8000/api/v1/repositories/repo3/datasets/dataset1/datapoints",
        headers={
            "Authorization": "Bearer some-token",
        },
        stream=True,
    )


def test_do_request(data_client: DataClient, mock_session: Mock) -> None:
    mock_response = Response()
    mock_response.status_code = 200
    mock_session.request.return_value = mock_response

    response = data_client._do_request("GET", "https://example.com")

    assert response == mock_response

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
    mock_session.request.side_effect = RequestException("Request failed")

    with pytest.raises(DataInternalError):
        data_client._do_request("GET", "https://example.com")

    mock_session.request.assert_called_once()


def test_do_request_handles_status_code_exception(
    data_client: DataClient, mock_session: Mock
) -> None:
    mock_response = Response()
    mock_response.status_code = 404
    mock_session.request.return_value = mock_response

    with pytest.raises(DataResourceNotFound):
        data_client._do_request("GET", "https://example.com")

    mock_session.request.assert_called_once()


def test_create_stage(data_client: DataClient, mock_session: Mock) -> None:
    def return_json_override() -> dict[Any, Any]:
        return {
            "stageId": "stage1",
            "name": "Stage 1",
            "createdAt": "2022-01-01T00:00:00Z",
            "updatedAt": "2022-01-01T00:00:00Z",
        }

    mock_response = Mock(spec=Response)
    mock_response.status_code = 201
    mock_response.json.return_value = return_json_override()
    mock_session.request.return_value = mock_response

    stage = data_client.create_stage(
        stage=DataStageCreate(
            name="Stage 1",
        )
    )

    assert isinstance(stage, DataStage)
    assert stage.stage_id == "stage1"
    assert stage.name == "Stage 1"

    mock_session.request.assert_called_once_with(
        "POST",
        "http://localhost:8000/api/v1/stages",
        headers={
            "Authorization": "Bearer some-token",
        },
        json={
            "name": "Stage 1",
        },
    )


def test_create_stage_handles_request_exception(
    data_client: DataClient, mock_session: Mock
) -> None:
    mock_session.request.side_effect = RequestException("Request failed")

    with pytest.raises(DataInternalError):
        data_client.create_stage(
            stage=DataStageCreate(
                name="Stage 1",
            )
        )

    mock_session.request.assert_called_once()


def test_list_stages(data_client: DataClient, mock_session: Mock) -> None:
    def return_json_override() -> dict[Any, Any]:
        return {
            "stages": [
                {
                    "stageId": "stage1",
                    "name": "Stage 1",
                    "createdAt": "2022-01-01T00:00:00Z",
                    "updatedAt": "2022-01-01T00:00:00Z",
                },
                {
                    "stageId": "stage2",
                    "name": "Stage 2",
                    "createdAt": "2022-01-01T00:00:00Z",
                    "updatedAt": "2022-01-01T00:00:00Z",
                },
            ]
        }

    mock_response = Mock(spec=Response)
    mock_response.status_code = 200
    mock_response.json.return_value = return_json_override()
    mock_session.request.return_value = mock_response

    stages = data_client.list_stages()

    assert len(stages) == 2
    assert isinstance(stages[0], DataStage)
    assert stages[0].stage_id == "stage1"
    assert stages[0].name == "Stage 1"
    assert isinstance(stages[1], DataStage)
    assert stages[1].stage_id == "stage2"
    assert stages[1].name == "Stage 2"

    mock_session.request.assert_called_once_with(
        "GET",
        "http://localhost:8000/api/v1/stages?page=0&size=20",
        headers={
            "Authorization": "Bearer some-token",
        },
    )


def test_list_stages_handles_request_exception(
    data_client: DataClient, mock_session: Mock
) -> None:
    mock_session.request.side_effect = RequestException("Request failed")

    with pytest.raises(DataInternalError):
        data_client.list_stages()

    mock_session.request.assert_called_once()


def test_get_stage(data_client: DataClient, mock_session: Mock) -> None:
    def return_json_override() -> dict[Any, Any]:
        return {
            "stageId": "stage1",
            "name": "Stage 1",
            "createdAt": "2022-01-01T00:00:00Z",
            "updatedAt": "2022-01-01T00:00:00Z",
        }

    mock_response = Mock(spec=Response)
    mock_response.status_code = 200
    mock_response.json.return_value = return_json_override()
    mock_session.request.return_value = mock_response

    stage = data_client.get_stage(stage_id="stage1")

    assert isinstance(stage, DataStage)
    assert stage.stage_id == "stage1"
    assert stage.name == "Stage 1"

    mock_session.request.assert_called_once_with(
        "GET",
        "http://localhost:8000/api/v1/stages/stage1",
        headers={
            "Authorization": "Bearer some-token",
        },
    )


def test_get_stage_handles_request_exception(
    data_client: DataClient, mock_session: Mock
) -> None:
    mock_session.request.side_effect = RequestException("Request failed")

    with pytest.raises(DataInternalError):
        data_client.get_stage(stage_id="stage1")

    mock_session.request.assert_called_once()

def test_upload_file_to_stage(data_client: DataClient, mock_session: Mock) -> None:
    def return_json_override() -> dict[Any, Any]:
        return {
            "fileId": "file1",
            "stageId": "stage1",
            "name": "File 1",
            "mediaType": "text/plain",
            "size": 100,
            "createdAt": "2022-01-01T00:00:00Z",
            "updatedAt": "2022-01-01T00:00:00Z",
        }

    mock_response = Mock(spec=Response)
    mock_response.status_code = 201
    mock_response.json.return_value = return_json_override()
    mock_session.request.return_value = mock_response

    file = data_client.upload_file_to_stage(
        stage_id="stage1",
        file=DataFileCreate(
            source_data=b"source_data",
            name="File 1",
        ),
    )

    assert isinstance(file, DataFile)
    assert file.file_id == "file1"
    assert file.stage_id == "stage1"
    assert file.name == "File 1"
    assert file.media_type == "text/plain"
    assert file.size == 100

    mock_session.request.assert_called_once_with(
        "POST",
        "http://localhost:8000/api/v1/stages/stage1/files",
        headers={
            "Authorization": "Bearer some-token",
        },
        files={
            "sourceData": b"source_data",
            "name": "File 1",
        },
    )

def test_upload_file_to_stage_handle_request_exception(
    data_client: DataClient, mock_session: Mock
) -> None:
    mock_session.request.side_effect = RequestException("Request failed")

    with pytest.raises(DataInternalError):
        data_client.upload_file_to_stage(
            stage_id="stage1",
            file=DataFileCreate(
                source_data=b"source_data",
                name="File 1",
            ),
        )

    mock_session.request.assert_called_once()

def test_list_files_in_stage(data_client: DataClient, mock_session: Mock) -> None:
    def return_json_override() -> dict[Any, Any]:
        return {
            "files": [
                {
                    "fileId": "file1",
                    "stageId": "stage1",
                    "name": "File 1",
                    "mediaType": "text/plain",
                    "size": 100,
                    "createdAt": "2022-01-01T00:00:00Z",
                    "updatedAt": "2022-01-01T00:00:00Z",
                },
                {
                    "fileId": "file2",
                    "stageId": "stage1",
                    "name": "File 2",
                    "mediaType": "text/plain",
                    "size": 200,
                    "createdAt": "2022-01-01T00:00:00Z",
                    "updatedAt": "2022-01-01T00:00:00Z",
                },
            ]
        }

    mock_response = Mock(spec=Response)
    mock_response.status_code = 200
    mock_response.json.return_value = return_json_override()
    mock_session.request.return_value = mock_response

    files = data_client.list_files_in_stage(stage_id="stage1")

    assert len(files) == 2
    assert isinstance(files[0], DataFile)
    assert files[0].file_id == "file1"
    assert files[0].stage_id == "stage1"
    assert files[0].name == "File 1"
    assert files[0].media_type == "text/plain"
    assert files[0].size == 100
    assert isinstance(files[1], DataFile)
    assert files[1].file_id == "file2"
    assert files[1].stage_id == "stage1"
    assert files[1].name == "File 2"
    assert files[1].media_type == "text/plain"
    assert files[1].size == 200

    mock_session.request.assert_called_once_with(
        "GET",
        "http://localhost:8000/api/v1/stages/stage1/files?page=0&size=20",
        headers={
            "Authorization": "Bearer some-token",
        },
    )

def test_list_files_in_stage_handle_request_exception(
    data_client: DataClient, mock_session: Mock
) -> None:
    mock_session.request.side_effect = RequestException("Request failed")

    with pytest.raises(DataInternalError):
        data_client.list_files_in_stage(stage_id="stage1")

    mock_session.request.assert_called_once()

def test_get_file_from_stage(data_client: DataClient, mock_session: Mock) -> None:
    expected_data = b"expected file content"

    def mock_file(*args: Any, **kwargs: Any) -> bytes:
        return b"expected file content"
    
    mock_response = Mock(spec=Response)
    mock_response.status_code = 200
    mock_response.content = mock_file()
    mock_session.request.return_value = mock_response

    file_content = data_client.get_file_from_stage(stage_id="stage1", file_id="file1")

    assert file_content.getvalue() == expected_data

    mock_session.request.assert_called_once_with(
        "GET",
        "http://localhost:8000/api/v1/stages/stage1/files/file1",
        headers={
            "Authorization": "Bearer some-token",
        },
    )

def test_get_file_from_stage_handles_request_exception(
    data_client: DataClient, mock_session: Mock
) -> None:
    mock_session.request.side_effect = RequestException("Request failed")

    with pytest.raises(DataInternalError):
        data_client.get_file_from_stage(stage_id="stage1", file_id="file1")

    mock_session.request.assert_called_once()