from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from fsspec import AbstractFileSystem  # type: ignore
from pydantic import BaseModel

from intelligence_layer.connectors.data.data import DataClient
from intelligence_layer.evaluation.aggregation.domain import AggregationOverview
from intelligence_layer.evaluation.aggregation.studio_aggregation_repository import (
    StudioAggregationRepository,
)
from intelligence_layer.evaluation.dataset.studio_dataset_repository import (
    StudioDatasetRepository,
)
from intelligence_layer.evaluation.evaluation.domain import EvaluationOverview
from intelligence_layer.evaluation.run.domain import RunOverview


class MockFileSystem(AbstractFileSystem):  # type: ignore
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


def test_upload_aggregation_overview_to_studio_repository(
    mock_studio_dataset_repository: StudioDatasetRepository, mock_data_client: Mock
) -> None:
    mock_studio_dataset_repository.create_dataset = Mock()  # type: ignore
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

    class Evaluation(BaseModel):
        evaluation_id: str
        example_id: str
        result: Any

    aggregation_overview = AggregationOverview[Evaluation](
        evaluation_overviews=frozenset([evaluation_overview]),
        id="aggregation1",
        start=datetime.now(),
        end=datetime.now(),
        crashed_during_evaluation_count=0,
        successful_evaluation_count=1,
        statistics=Evaluation(
            evaluation_id="evaluation1", example_id="example1", result="result1"
        ),
        description="description",
        metadata={"metadata1": "value1", "metadata2": "value2"},
        labels={"label1", "label2"},
    )

    # Create an instance of the StudioRunnerRepository
    studio_evaluation_repository = StudioAggregationRepository(
        file_system=MockFileSystem(),
        root_directory=Path("/aggregation"),
        studio_dataset_repository=mock_studio_dataset_repository,
    )
    # Call the store_run_overview method
    studio_evaluation_repository.store_aggregation_overview(aggregation_overview)

    # Assert that the create_dataset method was called with the correct arguments
    mock_studio_dataset_repository.create_dataset.assert_called_once_with(
        examples=[aggregation_overview],
        dataset_name=aggregation_overview.id,
        labels=aggregation_overview.labels.union(set([aggregation_overview.id])),
        metadata={"aggregation_id": aggregation_overview.id},
    )
