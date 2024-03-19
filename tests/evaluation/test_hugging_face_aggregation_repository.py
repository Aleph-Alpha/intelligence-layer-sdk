from pathlib import Path
from typing import Iterable
from uuid import uuid4

import huggingface_hub  # type: ignore
from _pytest.fixtures import fixture

from intelligence_layer.core import utc_now
from intelligence_layer.evaluation import (
    AggregationOverview,
    HuggingFaceAggregationRepository,
)
from intelligence_layer.evaluation.aggregation.file_aggregation_repository import (
    FileSystemAggregationRepository,
)
from tests.evaluation.conftest import DummyAggregatedEvaluation


@fixture
def dummy_aggregated_evaluation() -> DummyAggregatedEvaluation:
    return DummyAggregatedEvaluation(score=0.5)


@fixture(scope="session")
def hugging_face_aggregation_repository_id() -> str:
    return "Aleph-Alpha/test-aggregation-datasets"


@fixture(scope="session")
def hugging_face_aggregation_repository(
    hugging_face_token: str, hugging_face_aggregation_repository_id: str
) -> Iterable[HuggingFaceAggregationRepository]:
    try:
        yield HuggingFaceAggregationRepository(
            hugging_face_aggregation_repository_id,
            token=hugging_face_token,
            private=True,
        )
    finally:
        huggingface_hub.delete_repo(
            repo_id=hugging_face_aggregation_repository_id,
            token=hugging_face_token,
            repo_type="dataset",
            missing_ok=True,
        )


@fixture
def aggregation_overview(
    dummy_aggregated_evaluation: DummyAggregatedEvaluation,
) -> AggregationOverview[DummyAggregatedEvaluation]:
    return AggregationOverview(
        evaluation_overviews=frozenset([]),
        id=str(uuid4()),
        start=utc_now(),
        end=utc_now(),
        successful_evaluation_count=0,
        crashed_during_evaluation_count=0,
        description="",
        statistics=dummy_aggregated_evaluation,
    )


def test_repository_operations(
    hugging_face_aggregation_repository: HuggingFaceAggregationRepository,
    aggregation_overview: AggregationOverview[DummyAggregatedEvaluation],
) -> None:
    hugging_face_aggregation_repository.store_aggregation_overview(aggregation_overview)
    overview = hugging_face_aggregation_repository.aggregation_overview(
        aggregation_overview.id, DummyAggregatedEvaluation
    )

    assert aggregation_overview.id in list(
        hugging_face_aggregation_repository.aggregation_overview_ids()
    )
    assert overview != []


def test_file_exists_in_subdirectory(
    hugging_face_aggregation_repository: HuggingFaceAggregationRepository,
    aggregation_overview: AggregationOverview[DummyAggregatedEvaluation],
    hugging_face_aggregation_repository_id: str,
) -> None:
    hugging_face_aggregation_repository.store_aggregation_overview(aggregation_overview)
    path_to_file = Path(
        f"{hugging_face_aggregation_repository_id}/{FileSystemAggregationRepository._SUB_DIRECTORY}/{aggregation_overview.id}.json"
    )
    file_exists = hugging_face_aggregation_repository.exists(path_to_file)
    assert file_exists
