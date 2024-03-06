from typing import Iterable
from uuid import uuid4

from _pytest.fixtures import FixtureRequest
from pytest import fixture, mark

from intelligence_layer.core import utc_now
from intelligence_layer.evaluation import AggregationOverview, AggregationRepository
from intelligence_layer.evaluation.domain import EvaluationOverview
from tests.evaluation.conftest import DummyAggregatedEvaluation

test_repository_fixtures = [
    "file_aggregation_repository",
    "in_memory_aggregation_repository",
]


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
def test_aggregation_overview_returns_none_for_not_existing_id(
    repository_fixture: str,
    request: FixtureRequest,
    aggregation_overview: AggregationOverview[DummyAggregatedEvaluation],
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
    evaluation_overview: EvaluationOverview,
    aggregation_overviews: Iterable[AggregationOverview[DummyAggregatedEvaluation]],
    dummy_aggregated_evaluation: DummyAggregatedEvaluation,
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
    evaluation_overview: EvaluationOverview,
    aggregation_overviews: Iterable[AggregationOverview[DummyAggregatedEvaluation]],
    dummy_aggregated_evaluation: DummyAggregatedEvaluation,
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
