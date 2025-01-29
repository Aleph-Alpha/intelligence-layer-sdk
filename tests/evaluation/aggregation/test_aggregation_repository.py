from collections.abc import Iterable
from unittest.mock import patch
from uuid import uuid4

import pytest
from _pytest.fixtures import FixtureRequest
from fsspec.implementations.memory import MemoryFileSystem  # type: ignore
from pydantic import BaseModel, ValidationError
from pytest import fixture, mark

from intelligence_layer.core import utc_now
from intelligence_layer.evaluation import (
    AggregationOverview,
    AggregationRepository,
    EvaluationOverview,
)
from intelligence_layer.evaluation.aggregation.hugging_face_aggregation_repository import (
    HuggingFaceAggregationRepository,
)
from tests.evaluation.conftest import DummyAggregatedEvaluation

test_repository_fixtures = [
    "file_aggregation_repository",
    "in_memory_aggregation_repository",
    "mocked_hugging_face_aggregation_repository",
]


@fixture
def mocked_hugging_face_aggregation_repository(
    temp_file_system: MemoryFileSystem,
) -> Iterable[HuggingFaceAggregationRepository]:
    class_to_patch = "intelligence_layer.evaluation.aggregation.hugging_face_aggregation_repository.HuggingFaceAggregationRepository"
    with (
        patch(f"{class_to_patch}.create_repository", autospec=True),
        patch(
            f"{class_to_patch}.delete_repository",
            autospec=True,
        ),
    ):
        repo = HuggingFaceAggregationRepository(
            repository_id="doesn't-matter",
            token="non-existing-token",
            private=True,
        )
        repo._file_system = temp_file_system
        yield repo


@fixture
def aggregation_overviews(
    evaluation_overview: EvaluationOverview,
    dummy_aggregated_evaluation: DummyAggregatedEvaluation,
) -> Iterable[AggregationOverview[DummyAggregatedEvaluation]]:
    aggregation_ids = [str(uuid4()) for _ in range(10)]
    aggregation_overviews: list[AggregationOverview[DummyAggregatedEvaluation]] = []
    for aggregation_id in aggregation_ids:
        now = utc_now()
        aggregation_overviews.append(
            AggregationOverview(
                id=aggregation_id,
                evaluation_overviews=frozenset([evaluation_overview]),
                start=now,
                end=now,
                successful_evaluation_count=5,
                crashed_during_evaluation_count=3,
                statistics=dummy_aggregated_evaluation,
                description="dummy-aggregator",
            )
        )
    return aggregation_overviews


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_aggregation_repository_stores_and_returns_an_aggregation_overview(
    repository_fixture: str,
    request: FixtureRequest,
    aggregation_overview: AggregationOverview[DummyAggregatedEvaluation],
) -> None:
    aggregation_repository: AggregationRepository = request.getfixturevalue(
        repository_fixture
    )

    aggregation_repository.store_aggregation_overview(aggregation_overview)
    stored_aggregation_overview = aggregation_repository.aggregation_overview(
        aggregation_overview.id, DummyAggregatedEvaluation
    )

    assert stored_aggregation_overview == aggregation_overview


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_aggregation_overview_does_not_work_with_incorrect_types(
    repository_fixture: str,
    request: FixtureRequest,
    aggregation_overview: AggregationOverview[DummyAggregatedEvaluation],
) -> None:
    class InvalidClass(BaseModel):
        data: str

    aggregation_repository: AggregationRepository = request.getfixturevalue(
        repository_fixture
    )

    aggregation_repository.store_aggregation_overview(aggregation_overview)
    with pytest.raises(ValidationError):
        aggregation_repository.aggregation_overview(
            aggregation_overview.id, InvalidClass
        )
    with pytest.raises(ValidationError):
        next(iter(aggregation_repository.aggregation_overviews(InvalidClass)))


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_aggregation_overview_returns_none_for_not_existing_id(
    repository_fixture: str,
    request: FixtureRequest,
) -> None:
    aggregation_repository: AggregationRepository = request.getfixturevalue(
        repository_fixture
    )

    stored_aggregation_overview = aggregation_repository.aggregation_overview(
        "not-existing-id", DummyAggregatedEvaluation
    )

    assert stored_aggregation_overview is None


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_aggregation_overviews_returns_all_aggregation_overviews(
    repository_fixture: str,
    request: FixtureRequest,
    aggregation_overviews: Iterable[AggregationOverview[DummyAggregatedEvaluation]],
) -> None:
    aggregation_repository: AggregationRepository = request.getfixturevalue(
        repository_fixture
    )
    for aggregation_overview in aggregation_overviews:
        aggregation_repository.store_aggregation_overview(aggregation_overview)

    stored_aggregation_overviews = list(
        aggregation_repository.aggregation_overviews(DummyAggregatedEvaluation)
    )

    assert stored_aggregation_overviews == sorted(
        aggregation_overviews, key=lambda overview: overview.id
    )


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_aggregation_overview_ids_returns_sorted_ids(
    repository_fixture: str,
    request: FixtureRequest,
    aggregation_overviews: Iterable[AggregationOverview[DummyAggregatedEvaluation]],
) -> None:
    aggregation_repository: AggregationRepository = request.getfixturevalue(
        repository_fixture
    )
    for aggregation_overview in aggregation_overviews:
        aggregation_repository.store_aggregation_overview(aggregation_overview)

    stored_aggregation_ids = list(aggregation_repository.aggregation_overview_ids())

    assert stored_aggregation_ids == sorted(
        [aggregation_overview.id for aggregation_overview in aggregation_overviews]
    )
