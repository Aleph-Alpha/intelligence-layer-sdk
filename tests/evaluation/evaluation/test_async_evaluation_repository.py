from collections.abc import Iterable
from pathlib import Path
from uuid import uuid4

from pytest import FixtureRequest, fixture, mark

from intelligence_layer.core.tracer.tracer import utc_now
from intelligence_layer.evaluation import (
    AsyncEvaluationRepository,
    AsyncFileEvaluationRepository,
    RunOverview,
)
from intelligence_layer.evaluation.evaluation.domain import (
    EvaluationOverview,
    PartialEvaluationOverview,
)


@fixture
def async_file_evaluation_repository(tmp_path: Path) -> AsyncFileEvaluationRepository:
    return AsyncFileEvaluationRepository(tmp_path)


test_repository_fixtures = [
    "async_file_evaluation_repository",
    "async_in_memory_evaluation_repository",
]


@fixture
def partial_evaluation_overviews(
    run_overview: RunOverview,
) -> Iterable[PartialEvaluationOverview]:
    evaluation_ids = [str(uuid4()) for _ in range(10)]
    evaluation_overviews = []
    for evaluation_id in evaluation_ids:
        evaluation_overviews.append(
            PartialEvaluationOverview(
                id=evaluation_id,
                start_date=utc_now(),
                run_overviews=frozenset([run_overview]),
                submitted_evaluation_count=10,
                description="test evaluation overview",
                labels=set(),
                metadata=dict(),
            )
        )
    return evaluation_overviews


@fixture
def partial_evaluation_overview(
    evaluation_id: str, run_overview: RunOverview
) -> PartialEvaluationOverview:
    return PartialEvaluationOverview(
        id=evaluation_id,
        start_date=utc_now(),
        run_overviews=frozenset([run_overview]),
        submitted_evaluation_count=10,
        description="test evaluation overview",
        labels=set(),
        metadata=dict(),
    )


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_store_partial_evaluation_overview_stores_and_returns_given_evaluation_overview(
    repository_fixture: str,
    request: FixtureRequest,
    partial_evaluation_overview: PartialEvaluationOverview,
) -> None:
    evaluation_repository: AsyncEvaluationRepository = request.getfixturevalue(
        repository_fixture
    )

    evaluation_repository.store_partial_evaluation_overview(partial_evaluation_overview)
    retrieved_evaluation_overview = evaluation_repository.partial_evaluation_overview(
        partial_evaluation_overview.id
    )

    assert retrieved_evaluation_overview == partial_evaluation_overview


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_partial_evaluation_overview_returns_none_for_a_not_existing_overview_id(
    repository_fixture: str,
    request: FixtureRequest,
) -> None:
    evaluation_repository: AsyncEvaluationRepository = request.getfixturevalue(
        repository_fixture
    )

    evaluation_overview = evaluation_repository.partial_evaluation_overview(
        "not-existing-evaluation-id"
    )

    assert evaluation_overview is None


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_partial_evaluation_overviews_returns_all_evaluation_overviews(
    repository_fixture: str,
    request: FixtureRequest,
    partial_evaluation_overviews: Iterable[PartialEvaluationOverview],
) -> None:
    evaluation_repository: AsyncEvaluationRepository = request.getfixturevalue(
        repository_fixture
    )
    for evaluation_overview in partial_evaluation_overviews:
        evaluation_repository.store_partial_evaluation_overview(evaluation_overview)

    stored_evaluation_overviews = list(
        evaluation_repository.partial_evaluation_overviews()
    )

    assert stored_evaluation_overviews == sorted(
        partial_evaluation_overviews, key=lambda eval_overview: eval_overview.id
    )


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_partial_and_full_evaluation_overview_dont_overlap(
    repository_fixture: str,
    request: FixtureRequest,
    partial_evaluation_overview: PartialEvaluationOverview,
    evaluation_overview: EvaluationOverview,
) -> None:
    evaluation_repository: AsyncEvaluationRepository = request.getfixturevalue(
        repository_fixture
    )

    evaluation_repository.store_partial_evaluation_overview(partial_evaluation_overview)
    evaluation_repository.store_evaluation_overview(evaluation_overview)

    retrieved_partial_evaluation_overview = (
        evaluation_repository.partial_evaluation_overview(
            partial_evaluation_overview.id
        )
    )
    retrieved_evaluation_overview = evaluation_repository.evaluation_overview(
        partial_evaluation_overview.id
    )

    all_partial_overviews = list(evaluation_repository.partial_evaluation_overviews())
    all_full_overviews = list(evaluation_repository.evaluation_overviews())

    assert retrieved_partial_evaluation_overview == partial_evaluation_overview
    assert retrieved_evaluation_overview == evaluation_overview

    assert len(all_partial_overviews) == 1
    assert len(all_full_overviews) == 1

    assert all_partial_overviews[0] == partial_evaluation_overview
    assert all_full_overviews[0] == evaluation_overview


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_can_retrieve_failed_evaluations_for_partial_evaluations(
    repository_fixture: str,
    request: FixtureRequest,
    partial_evaluation_overview: PartialEvaluationOverview,
) -> None:
    evaluation_repository: AsyncEvaluationRepository = request.getfixturevalue(
        repository_fixture
    )
    some_dummy_type = PartialEvaluationOverview

    evaluation_repository.store_partial_evaluation_overview(partial_evaluation_overview)
    n_failed = len(
        evaluation_repository.failed_example_evaluations(
            partial_evaluation_overview.id, some_dummy_type
        )
    )

    assert n_failed == 0
