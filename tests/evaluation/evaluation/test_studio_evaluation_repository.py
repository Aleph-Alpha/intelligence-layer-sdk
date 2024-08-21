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
from intelligence_layer.evaluation.evaluation.domain import EvaluationOverview
from intelligence_layer.evaluation.evaluation.studio_evaluation_repository import StudioEvaluationRepository
from intelligence_layer.evaluation.run.domain import RunOverview
from intelligence_layer.evaluation.run.file_run_repository import FileSystemRunRepository
from intelligence_layer.evaluation.run.studio_runner_repository import StudioRunnerRepository
from intelligence_layer.core import Output

class MockFileSystem(AbstractFileSystem): # type: ignore
    pass


class MockEvaluationOutput(BaseModel):
    evaluation_id: str
    example_id: str
    result: Any

class MockExampleOutput(BaseModel):
    output: Any

@pytest.fixture
def mock_data_client() -> Mock:
    return Mock(spec=DataClient, base_data_platform_url="http://localhost:3000")


@pytest.fixture
def mock_studio_dataset_repository(mock_data_client: Mock) -> StudioDatasetRepository:
    return StudioDatasetRepository(repository_id="repo1", data_client=mock_data_client)


def test_upload_evaluation_to_studio(mock_studio_dataset_repository: StudioDatasetRepository, mock_data_client: Mock)-> None:

    mock_studio_dataset_repository.create_dataset = Mock() # type: ignore
    mock_studio_dataset_repository.create_dataset.return_value = Mock(
        id="dataset_id",
        labels={"label1", "label2"},
        metadata={},
        name="Dataset 1",
    )
    
    # Create a mock overview
    run_overview = RunOverview(
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

    evaluation_overview = EvaluationOverview(
        run_overviews=frozenset([run_overview]),
        id="evaluation1",
        start_date=datetime.now(),
        end_date=datetime.now(),
        failed_evaluation_count=0,
        successful_evaluation_count=1,
        description="description",
        metadata={"metadata1": "value1", "metadata2": "value2"},
        labels={"label1", "label2"},
    )

    # Create an instance of the StudioRunnerRepository
    studio_evaluation_repository = StudioEvaluationRepository(
        file_system=MockFileSystem(),
        root_directory=Path("/root"),
        evaluation_type=MockEvaluationOutput,
        studio_dataset_repository=mock_studio_dataset_repository,
    )

    studio_evaluation_repository.example_evaluations = Mock() # type: ignore
    studio_evaluation_repository.example_evaluations.return_value = [
        MockEvaluationOutput(example_id="example1", evaluation_id="evaluation1", result=MockExampleOutput(output="output1")),
        MockEvaluationOutput(example_id="example2", evaluation_id="evaluation1", result=MockExampleOutput(output="output2")),
    ]

    # Call the store_run_overview method
    studio_evaluation_repository.store_evaluation_overview(evaluation_overview)

    # Assert that the create_dataset method was called with the correct arguments
    mock_studio_dataset_repository.create_dataset.assert_called_once_with(
        examples=studio_evaluation_repository.example_evaluations.return_value,
        dataset_name=evaluation_overview.id,
        labels=evaluation_overview.labels.union(set([evaluation_overview.id])),
        metadata=evaluation_overview.model_dump(mode='json'),
    )
    