from collections.abc import Iterable
from itertools import chain
from uuid import uuid4

import pytest
from _pytest.fixtures import FixtureRequest
from pydantic import BaseModel, ValidationError
from pytest import fixture, mark

from intelligence_layer.core import utc_now
from intelligence_layer.evaluation import (
    Evaluation,
    EvaluationOverview,
    EvaluationRepository,
    ExampleEvaluation,
    FailedExampleEvaluation,
    RunOverview,
)
from tests.evaluation.conftest import DummyEvaluation

test_repository_fixtures = [
    "file_evaluation_repository",
    "in_memory_evaluation_repository",
]


@fixture
def successful_example_evaluations(
    evaluation_id: str,
) -> Iterable[ExampleEvaluation[DummyEvaluation]]:
    example_ids = [str(uuid4()) for _ in range(10)]
    example_evaluations = []
    for example_id in example_ids:
        example_evaluations.append(
            ExampleEvaluation(
                evaluation_id=evaluation_id,
                example_id=example_id,
                result=DummyEvaluation(result="result"),
            )
        )
    return example_evaluations


@fixture
def failed_example_evaluations(
    evaluation_id: str,
) -> Iterable[ExampleEvaluation[DummyEvaluation]]:
    example_ids = [str(uuid4()) for _ in range(10)]
    example_evaluations: list[ExampleEvaluation[DummyEvaluation]] = []
    for example_id in example_ids:
        example_evaluations.append(
            ExampleEvaluation(
                evaluation_id=evaluation_id,
                example_id=example_id,
                result=FailedExampleEvaluation(error_message="some error"),
            )
        )
    return example_evaluations


@fixture
def evaluation_overviews(run_overview: RunOverview) -> Iterable[EvaluationOverview]:
    evaluation_ids = [str(uuid4()) for _ in range(10)]
    evaluation_overviews = []
    for evaluation_id in evaluation_ids:
        evaluation_overviews.append(
            EvaluationOverview(
                id=evaluation_id,
                start_date=utc_now(),
                end_date=utc_now(),
                successful_evaluation_count=1,
                failed_evaluation_count=1,
                run_overviews=frozenset([run_overview]),
                description="test evaluation overview 1",
                labels=set(),
                metadata={},
            )
        )
    return evaluation_overviews


def get_example_via_both_retrieval_methods(
    evaluation_repository: EvaluationRepository,
    run_id: str,
    example_id: str,
    output_type: type[Evaluation],
) -> Iterable[
    ExampleEvaluation[Evaluation] | ExampleEvaluation[FailedExampleEvaluation] | None
]:
    yield evaluation_repository.example_evaluation(run_id, example_id, output_type)
    yield next(iter(evaluation_repository.example_evaluations(run_id, output_type)))


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_create_evaluation_dataset_returns_an_evaluation_dataset_id(
    repository_fixture: str,
    request: FixtureRequest,
) -> None:
    evaluation_repository: EvaluationRepository = request.getfixturevalue(
        repository_fixture
    )

    evaluation_id = evaluation_repository.initialize_evaluation()

    assert len(evaluation_id) > 0


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_evaluation_overview_ids_returns_all_sorted_ids(
    repository_fixture: str,
    request: FixtureRequest,
    run_overview: RunOverview,
    evaluation_overviews: Iterable[EvaluationOverview],
) -> None:
    evaluation_repository: EvaluationRepository = request.getfixturevalue(
        repository_fixture
    )
    for evaluation_overview in evaluation_overviews:
        evaluation_repository.store_evaluation_overview(evaluation_overview)

    stored_evaluation_ids = list(evaluation_repository.evaluation_overview_ids())

    assert stored_evaluation_ids == sorted(
        [evaluation_overview.id for evaluation_overview in evaluation_overviews]
    )


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_evaluation_repository_stores_and_returns_an_example_evaluation(
    repository_fixture: str,
    request: FixtureRequest,
    successful_example_evaluation: ExampleEvaluation[DummyEvaluation],
) -> None:
    evaluation_repository: EvaluationRepository = request.getfixturevalue(
        repository_fixture
    )

    evaluation_repository.store_example_evaluation(successful_example_evaluation)
    for example_evaluation in get_example_via_both_retrieval_methods(
        evaluation_repository,
        successful_example_evaluation.evaluation_id,
        successful_example_evaluation.example_id,
        DummyEvaluation,
    ):
        assert example_evaluation == successful_example_evaluation


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_example_evaluation_returns_none_if_example_id_does_not_exist(
    repository_fixture: str,
    request: FixtureRequest,
    successful_example_evaluation: ExampleEvaluation[DummyEvaluation],
) -> None:
    evaluation_repository: EvaluationRepository = request.getfixturevalue(
        repository_fixture
    )
    evaluation_repository.store_example_evaluation(successful_example_evaluation)

    stored_example_evaluation = evaluation_repository.example_evaluation(
        successful_example_evaluation.evaluation_id,
        "not-existing-example-id",
        DummyEvaluation,
    )

    assert stored_example_evaluation is None


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_example_evaluation_does_not_work_with_incorrect_types(
    repository_fixture: str,
    request: FixtureRequest,
    successful_example_evaluation: ExampleEvaluation[DummyEvaluation],
) -> None:
    class InvalidType(BaseModel):
        data: str

    evaluation_repository: EvaluationRepository = request.getfixturevalue(
        repository_fixture
    )
    evaluation_repository.store_example_evaluation(successful_example_evaluation)

    with pytest.raises(ValidationError):
        evaluation_repository.example_evaluation(
            successful_example_evaluation.evaluation_id,
            successful_example_evaluation.example_id,
            InvalidType,
        )

    with pytest.raises(ValidationError):
        evaluation_repository.example_evaluations(
            successful_example_evaluation.evaluation_id,
            InvalidType,
        )


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_example_evaluation_throws_if_evaluation_id_does_not_exist(
    repository_fixture: str,
    request: FixtureRequest,
    successful_example_evaluation: ExampleEvaluation[DummyEvaluation],
) -> None:
    evaluation_repository: EvaluationRepository = request.getfixturevalue(
        repository_fixture
    )
    evaluation_repository.store_example_evaluation(successful_example_evaluation)

    with pytest.raises(ValueError):
        evaluation_repository.example_evaluation(
            "not-existing-eval-id",
            successful_example_evaluation.example_id,
            DummyEvaluation,
        )
    with pytest.raises(ValueError):
        evaluation_repository.example_evaluation(
            "not-existing-eval-id",
            "not-existing-example-id",
            DummyEvaluation,
        )
    with pytest.raises(ValueError):
        evaluation_repository.example_evaluations(
            "not-existing-eval-id",
            DummyEvaluation,
        )


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_evaluation_repository_stores_and_returns_a_failed_example_evaluation(
    repository_fixture: str,
    request: FixtureRequest,
) -> None:
    evaluation_repository: EvaluationRepository = request.getfixturevalue(
        repository_fixture
    )
    evaluation_id = "evaluation-id"
    failed_example_evaluation: ExampleEvaluation[FailedExampleEvaluation] = (
        ExampleEvaluation(
            evaluation_id=evaluation_id,
            example_id="example-id",
            result=FailedExampleEvaluation(error_message="some error"),
        )
    )

    evaluation_repository.store_example_evaluation(failed_example_evaluation)
    for example_evaluation in get_example_via_both_retrieval_methods(
        evaluation_repository,
        evaluation_id,
        failed_example_evaluation.example_id,
        FailedExampleEvaluation,
    ):
        assert example_evaluation == failed_example_evaluation


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_example_evaluations_returns_all_example_evaluations(
    repository_fixture: str,
    request: FixtureRequest,
    successful_example_evaluations: Iterable[ExampleEvaluation[DummyEvaluation]],
    failed_example_evaluations: Iterable[ExampleEvaluation[DummyEvaluation]],
    evaluation_id: str,
) -> None:
    evaluation_repository: EvaluationRepository = request.getfixturevalue(
        repository_fixture
    )
    all_example_evaluations: list[ExampleEvaluation[DummyEvaluation]] = []

    for example_evaluation in chain(
        successful_example_evaluations, failed_example_evaluations
    ):
        evaluation_repository.store_example_evaluation(example_evaluation)
        all_example_evaluations.append(example_evaluation)

    example_evaluations = evaluation_repository.example_evaluations(
        evaluation_id, DummyEvaluation
    )

    assert list(example_evaluations) == sorted(
        all_example_evaluations,
        key=lambda evaluation: evaluation.example_id,
    )


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_successful_example_evaluations_returns_all_successful_example_evaluations(
    repository_fixture: str,
    request: FixtureRequest,
    evaluation_id: str,
    successful_example_evaluations: Iterable[ExampleEvaluation[DummyEvaluation]],
    failed_example_evaluations: Iterable[ExampleEvaluation[DummyEvaluation]],
) -> None:
    evaluation_repository: EvaluationRepository = request.getfixturevalue(
        repository_fixture
    )

    for example_evaluation in chain(
        successful_example_evaluations, failed_example_evaluations
    ):
        evaluation_repository.store_example_evaluation(example_evaluation)

    example_evaluations = evaluation_repository.successful_example_evaluations(
        evaluation_id, DummyEvaluation
    )

    assert list(example_evaluations) == sorted(
        successful_example_evaluations,
        key=lambda evaluation: evaluation.example_id,
    )


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_file_repository_can_fetch_failed_examples_from_evaluation_run(
    repository_fixture: str,
    request: FixtureRequest,
    evaluation_id: str,
    successful_example_evaluations: Iterable[ExampleEvaluation[DummyEvaluation]],
    failed_example_evaluations: Iterable[ExampleEvaluation[DummyEvaluation]],
) -> None:
    evaluation_repository = request.getfixturevalue(repository_fixture)
    for example_evaluation in chain(
        successful_example_evaluations, failed_example_evaluations
    ):
        evaluation_repository.store_example_evaluation(example_evaluation)
    example_evaluations = evaluation_repository.failed_example_evaluations(
        evaluation_id, DummyEvaluation
    )

    assert list(example_evaluations) == sorted(
        failed_example_evaluations,
        key=lambda evaluation: evaluation.example_id,
    )


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_store_evaluation_overview_stores_and_returns_given_evaluation_overview(
    repository_fixture: str,
    request: FixtureRequest,
    evaluation_overview: EvaluationOverview,
) -> None:
    evaluation_repository: EvaluationRepository = request.getfixturevalue(
        repository_fixture
    )

    evaluation_repository.store_evaluation_overview(evaluation_overview)
    retrieved_evaluation_overview = evaluation_repository.evaluation_overview(
        evaluation_overview.id
    )

    assert retrieved_evaluation_overview == evaluation_overview


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_can_retrieve_examples_and_failed_examples_after_storing_an_overview(
    repository_fixture: str,
    request: FixtureRequest,
    evaluation_overview: EvaluationOverview,
) -> None:
    some_dummy_type = EvaluationOverview

    evaluation_repository: EvaluationRepository = request.getfixturevalue(
        repository_fixture
    )

    evaluation_repository.store_evaluation_overview(evaluation_overview)

    n_failed_examples = len(
        evaluation_repository.failed_example_evaluations(
            evaluation_overview.id, some_dummy_type
        )
    )
    assert n_failed_examples == 0

    n_examples = len(
        evaluation_repository.example_evaluations(
            evaluation_overview.id, some_dummy_type
        )
    )
    assert n_examples == 0


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_evaluation_overview_returns_none_for_a_not_existing_overview_id(
    repository_fixture: str,
    request: FixtureRequest,
) -> None:
    evaluation_repository: EvaluationRepository = request.getfixturevalue(
        repository_fixture
    )

    evaluation_overview = evaluation_repository.evaluation_overview(
        "not-existing-evaluation-id"
    )

    assert evaluation_overview is None


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_evaluation_overviews_returns_all_evaluation_overviews(
    repository_fixture: str,
    request: FixtureRequest,
    run_overview: RunOverview,
    evaluation_overviews: Iterable[EvaluationOverview],
) -> None:
    evaluation_repository: EvaluationRepository = request.getfixturevalue(
        repository_fixture
    )
    for evaluation_overview in evaluation_overviews:
        evaluation_repository.store_evaluation_overview(evaluation_overview)

    stored_evaluation_overviews = list(evaluation_repository.evaluation_overviews())

    assert stored_evaluation_overviews == sorted(
        evaluation_overviews, key=lambda eval_overview: eval_overview.id
    )
