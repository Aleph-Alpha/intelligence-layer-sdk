from datetime import datetime
import os
from unittest.mock import Mock, patch
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
import pytest
from fsspec import AbstractFileSystem # type: ignore
from pydantic import BaseModel
from intelligence_layer.connectors.data.data import DataClient
from intelligence_layer.evaluation.dataset.studio_dataset_repository import StudioDatasetRepository
from intelligence_layer.evaluation.run.domain import RunOverview
from intelligence_layer.evaluation.run.file_run_repository import FileSystemRunRepository
from intelligence_layer.evaluation.run.studio_runner_repository import StudioRunnerRepository
from intelligence_layer.core import Output

class MockFileSystem(AbstractFileSystem): # type: ignore
    pass

class InputExample(BaseModel):
    data: str


class ExpectedOutputExample(BaseModel):
    data: str


class MockExampleOutput(BaseModel):
    example_id: str
    run_id: str
    output: Any

@pytest.fixture
def mock_data_client() -> Mock:
    return Mock(spec=DataClient, base_data_platform_url="http://localhost:3000")


@pytest.fixture
def mock_studio_data_repository(mock_data_client: Mock) -> StudioDatasetRepository:
    return StudioDatasetRepository(repository_id="repo1", data_client=mock_data_client)


def test_upload_to_studio_runner_repository(mock_studio_data_repository: StudioDatasetRepository, mock_data_client: Mock)-> None:

    mock_studio_data_repository.create_dataset = Mock() # type: ignore
    mock_studio_data_repository.create_dataset.return_value = Mock(
        id="dataset_id",
        labels={"label1", "label2"},
        metadata={},
        name="Dataset 1",
    )
    
    # Create a mock overview
    overview = RunOverview(
        id="run1",
        labels={"label1", "label2"},
        metadata={"metadata1": "value1", "metadata2": "value2"},
        start=datetime.now(),
        end=datetime.now(),
        failed_example_count=0,
        successful_example_count=2,
        description="description",
        dataset_id="dataset_id",
    )

    # Create an instance of the StudioRunnerRepository
    studio_runner_repository = StudioRunnerRepository(
        file_system=MockFileSystem(),
        root_directory=Path("/root"),
        output_type=MockExampleOutput,
        studio_dataset_repository=mock_studio_data_repository,
    )

    studio_runner_repository.example_outputs = Mock() # type: ignore
    studio_runner_repository.example_outputs.return_value = [
        MockExampleOutput(example_id="example1", run_id="run1", output="output1"),
        MockExampleOutput(example_id="example2", run_id="run1", output="output2"),
    ]

    # Call the store_run_overview method
    studio_runner_repository.store_run_overview(overview)

    # Assert that the create_dataset method was called with the correct arguments
    mock_studio_data_repository.create_dataset.assert_called_once_with(
        examples=studio_runner_repository.example_outputs.return_value,
        dataset_name=overview.id,
        labels=overview.labels.union(set([overview.id])),
        metadata=overview.model_dump(mode="json")
    )