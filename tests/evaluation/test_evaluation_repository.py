from datetime import datetime
from typing import Sequence
from uuid import uuid4

from _pytest.fixtures import FixtureRequest
from pydantic import BaseModel
from pytest import fixture, mark

from intelligence_layer.core import utc_now
from intelligence_layer.evaluation import (
    EvaluationRepository,
    ExampleEvaluation,
    ExampleTrace,
    FailedExampleEvaluation,
    FileEvaluationRepository,
    TaskSpanTrace,
)
from intelligence_layer.evaluation.domain import EvaluationOverview, RunOverview
from tests.evaluation.conftest import DummyEvaluation


class DummyEvaluationWithExceptionStructure(BaseModel):
    error_message: str


test_repository_fixtures = [
    "file_evaluation_repository",
    "in_memory_evaluation_repository",
]


@fixture
def task_span_trace() -> TaskSpanTrace:
    now = datetime.now()
    return TaskSpanTrace(
        name="task name", traces=[], start=now, end=now, input="input", output="output"
    )


@fixture
def example_trace(
    task_span_trace: TaskSpanTrace,
) -> ExampleTrace:
    return ExampleTrace(
        run_id="evaluation-id",
        example_id="example-id",
        trace=task_span_trace,
    )


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

    evaluation_id = evaluation_repository.create_evaluation_dataset()

    assert len(evaluation_id) > 0


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_evaluation_ids_returns_all_sorted_evaluation_ids(
    repository_fixture: str, request: FixtureRequest, run_overview: RunOverview
) -> None:
    evaluation_repository: EvaluationRepository = request.getfixturevalue(
        repository_fixture
    )
    evaluation_ids = [str(uuid4()) for _ in range(10)]
    for evaluation_id in evaluation_ids:
        evaluation_repository.store_evaluation_overview(
            EvaluationOverview(
                id=evaluation_id,
                start=utc_now(),
                run_overviews=frozenset([run_overview]),
                description="test evaluation overview 1",
            )
        )

    stored_evaluation_ids = list(evaluation_repository.evaluation_overview_ids())

    assert stored_evaluation_ids == sorted(evaluation_ids)


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_evaluation_repository_returns_none_in_case_example_result_does_not_exist(
    repository_fixture: str,
    request: FixtureRequest,
) -> None:
    evaluation_repository: EvaluationRepository = request.getfixturevalue(
        repository_fixture
    )

    example_evaluation = evaluation_repository.example_evaluation(
        "not-existing-evaluation-id",
        "not-existing-example-id",
        DummyEvaluation,
    )

    assert example_evaluation is None


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_evaluation_repository_stores_and_returns_an_example_evaluation(
    repository_fixture: str,
    request: FixtureRequest,
    successful_example_result: ExampleEvaluation[DummyEvaluation],
    evaluation_id: str,
) -> None:
    evaluation_repository: EvaluationRepository = request.getfixturevalue(
        repository_fixture
    )

    evaluation_repository.store_example_evaluation(successful_example_result)
    example_evaluation = evaluation_repository.example_evaluation(
        evaluation_id,
        successful_example_result.example_id,
        DummyEvaluation,
    )

    assert example_evaluation == successful_example_result


def test_evaluation_repository_stores_and_returns_a_failed_example_evaluation(
    file_evaluation_repository: FileEvaluationRepository,
) -> None:
    evaluation_id = "evaluation-id"
    failed_example_evaluation: ExampleEvaluation[FailedExampleEvaluation] = (
        ExampleEvaluation(
            evaluation_id=evaluation_id,
            example_id="example-id",
            result=FailedExampleEvaluation(error_message="some error"),
        )
    )

    file_evaluation_repository.store_example_evaluation(failed_example_evaluation)
    example_evaluation = file_evaluation_repository.example_evaluation(
        evaluation_id,
        failed_example_evaluation.example_id,
        DummyEvaluationWithExceptionStructure,
    )

    assert example_evaluation == failed_example_evaluation


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_example_evaluations_returns_an_empty_sequence_for_not_existing_evaluation_id(
    repository_fixture: str,
    request: FixtureRequest,
) -> None:
    evaluation_repository = request.getfixturevalue(repository_fixture)

    example_evaluations = evaluation_repository.example_evaluations(
        "not-existing-evaluation-id", DummyEvaluation
    )

    assert example_evaluations == []


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_example_evaluations_returns_all_example_evaluations(
    repository_fixture: str,
    request: FixtureRequest,
    successful_example_result: ExampleEvaluation[DummyEvaluation],
    failed_example_result: ExampleEvaluation[DummyEvaluation],
    evaluation_id: str,
) -> None:
    evaluation_repository = request.getfixturevalue(repository_fixture)
    successful_example_ids = [str(uuid4()) for _ in range(10)]
    for example_id in successful_example_ids:
        evaluation_repository.store_example_evaluation(
            ExampleEvaluation(
                evaluation_id=evaluation_id,
                example_id=example_id,
                result=DummyEvaluation(result="result"),
            )
        )
    failed_example_id = str(uuid4())
    evaluation_repository.store_example_evaluation(
        ExampleEvaluation(
            evaluation_id=evaluation_id,
            example_id=failed_example_id,
            result=FailedExampleEvaluation(error_message="error"),
        )
    )

    example_evaluations: Sequence[ExampleEvaluation[DummyEvaluation]] = (
        evaluation_repository.example_evaluations(evaluation_id, DummyEvaluation)
    )

    assert [
        example_evaluation.example_id for example_evaluation in example_evaluations
    ] == sorted(successful_example_ids + [failed_example_id])


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_file_repository_can_fetch_failed_examples_from_evaluation_run(
    repository_fixture: str,
    request: FixtureRequest,
    evaluation_id: str,
) -> None:
    evaluation_repository = request.getfixturevalue(repository_fixture)
    failed_example_ids = [str(uuid4()) for _ in range(10)]
    for example_id in failed_example_ids:
        evaluation_repository.store_example_evaluation(
            ExampleEvaluation(
                evaluation_id=evaluation_id,
                example_id=example_id,
                result=FailedExampleEvaluation(error_message="error"),
            )
        )
    successful_example_id = str(uuid4())
    evaluation_repository.store_example_evaluation(
        ExampleEvaluation(
            evaluation_id=evaluation_id,
            example_id=successful_example_id,
            result=DummyEvaluation(result="result"),
        )
    )

    example_evaluations = evaluation_repository.failed_example_evaluations(
        evaluation_id, DummyEvaluation
    )

    assert [
        example_evaluation.example_id for example_evaluation in example_evaluations
    ] == sorted(failed_example_ids)


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_failed_example_evaluations_returns_all_failed_example_evaluations(
    repository_fixture: str,
    request: FixtureRequest,
    successful_example_result: ExampleEvaluation[DummyEvaluation],
    failed_example_result: ExampleEvaluation[DummyEvaluation],
    evaluation_id: str,
) -> None:
    example_evaluations: Sequence[ExampleEvaluation[DummyEvaluation]] = [
        successful_example_result,
        failed_example_result,
    ]
    evaluation_repository = request.getfixturevalue(repository_fixture)
    for result in example_evaluations:
        evaluation_repository.store_example_evaluation(result)

    failed_example_evaluations = evaluation_repository.failed_example_evaluations(
        evaluation_id, DummyEvaluation
    )

    assert failed_example_evaluations == [failed_example_result]


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


def test_file_repository_returns_none_for_nonexisting_overview(
    file_evaluation_repository: FileEvaluationRepository,
) -> None:
    assert file_evaluation_repository.evaluation_overview("does-not-exist") is None
