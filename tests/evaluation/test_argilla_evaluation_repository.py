from itertools import chain
from typing import Iterable, Sequence, Tuple
from uuid import uuid4

from pytest import fixture

from intelligence_layer.connectors import ArgillaEvaluation, Field, Question, RecordData
from intelligence_layer.core import utc_now
from intelligence_layer.evaluation import (
    ArgillaEvaluationRepository,
    EvaluationOverview,
    ExampleEvaluation,
    FailedExampleEvaluation,
    InMemoryEvaluationRepository,
    RecordDataSequence,
    RunOverview,
)
from tests.evaluation.conftest import DummyEvaluation, StubArgillaClient


@fixture
def argilla_workspace_id() -> str:
    return "workspace-id"


@fixture
def argilla_client_fields() -> Sequence[Field]:
    return []


@fixture
def argilla_client_questions() -> Sequence[Question]:
    return []


@fixture
def stub_argilla_client_with_defaults(
    stub_argilla_client: StubArgillaClient,
    argilla_workspace_id: str,
    argilla_client_fields: Sequence[Field],
    argilla_client_questions: Sequence[Question],
) -> StubArgillaClient:
    stub_argilla_client._expected_workspace_id = argilla_workspace_id
    stub_argilla_client._expected_fields = argilla_client_fields
    stub_argilla_client._expected_questions = argilla_client_questions
    return stub_argilla_client


@fixture
def argilla_evaluation_repository(
    stub_argilla_client_with_defaults: StubArgillaClient,
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    argilla_workspace_id: str,
    argilla_client_fields: Sequence[Field],
    argilla_client_questions: Sequence[Question],
) -> ArgillaEvaluationRepository:
    return ArgillaEvaluationRepository(
        argilla_client=stub_argilla_client_with_defaults,
        evaluation_repository=in_memory_evaluation_repository,
        workspace_id=argilla_workspace_id,
        fields=argilla_client_fields,
        questions=argilla_client_questions,
    )


@fixture
def argilla_evaluation_repository_with_example_evaluations(
    argilla_evaluation_repository: ArgillaEvaluationRepository,
) -> Tuple[
    str,
    ArgillaEvaluationRepository,
    list[ExampleEvaluation[RecordDataSequence]],
    list[ExampleEvaluation[RecordDataSequence]],
]:
    dataset_id = argilla_evaluation_repository.initialize_evaluation()

    successful_example_evaluation_ids = [str(uuid4()) for _ in range(10)]

    successful_example_evaluations = []
    for example_evaluation_id in successful_example_evaluation_ids:
        example_evaluation = ExampleEvaluation(
            evaluation_id=dataset_id,
            example_id=example_evaluation_id,
            result=RecordDataSequence(
                records=[
                    RecordData(
                        content={}, example_id=example_evaluation_id, metadata={}
                    )
                ]
            ),
        )
        successful_example_evaluations.append(example_evaluation)
        argilla_evaluation_repository.store_example_evaluation(example_evaluation)

    failed_example_evaluation_ids = [str(uuid4()) for _ in range(10)]
    failed_example_evaluations = []
    for example_evaluation_id in failed_example_evaluation_ids:
        failed_example_evaluation: ExampleEvaluation[RecordDataSequence] = (
            ExampleEvaluation(
                evaluation_id=dataset_id,
                example_id=example_evaluation_id,
                result=FailedExampleEvaluation(error_message="error"),
            )
        )
        failed_example_evaluations.append(failed_example_evaluation)
        (
            argilla_evaluation_repository.store_example_evaluation(
                failed_example_evaluation
            )
        )

    return (
        dataset_id,
        argilla_evaluation_repository,
        successful_example_evaluations,
        failed_example_evaluations,
    )


@fixture
def evaluation_overviews(run_overview: RunOverview) -> Iterable[EvaluationOverview]:
    evaluation_overviews = []
    evaluation_ids = [str(uuid4()) for _ in range(10)]
    for evaluation_id in evaluation_ids:
        evaluation_overviews.append(
            EvaluationOverview(
                id=evaluation_id,
                start_date=utc_now(),
                end_date=utc_now(),
                successful_evaluation_count=3,
                failed_evaluation_count=2,
                skipped_evaluation_count=1,
                run_overviews=frozenset([run_overview]),
                description="test evaluation overview 1",
            )
        )

    return evaluation_overviews


def test_create_evaluation_dataset_returns_dataset_id(
    argilla_evaluation_repository: ArgillaEvaluationRepository,
) -> None:
    dataset_id = argilla_evaluation_repository.initialize_evaluation()

    assert dataset_id != ""


def test_evaluation_overview_returns_none_for_not_existing_id(
    argilla_evaluation_repository: ArgillaEvaluationRepository,
    evaluation_overview: EvaluationOverview,
) -> None:
    argilla_evaluation_repository.store_evaluation_overview(evaluation_overview)

    stored_evaluation_overview = argilla_evaluation_repository.evaluation_overview(
        "not-existing-id"
    )

    assert stored_evaluation_overview is None


def test_evaluation_overview_returns_evaluation_overview(
    argilla_evaluation_repository: ArgillaEvaluationRepository,
    evaluation_overview: EvaluationOverview,
) -> None:
    argilla_evaluation_repository.store_evaluation_overview(evaluation_overview)

    stored_evaluation_overview = argilla_evaluation_repository.evaluation_overview(
        evaluation_overview.id
    )

    assert stored_evaluation_overview == evaluation_overview


def test_evaluation_overviews_returns_sorted_evaluation_overviews(
    argilla_evaluation_repository: ArgillaEvaluationRepository,
    evaluation_overviews: Iterable[EvaluationOverview],
) -> None:
    for evaluation_overview in evaluation_overviews:
        argilla_evaluation_repository.store_evaluation_overview(evaluation_overview)

    stored_evaluation_overviews = list(
        argilla_evaluation_repository.evaluation_overviews()
    )

    assert stored_evaluation_overviews == sorted(
        evaluation_overviews, key=lambda overview: overview.id
    )


def test_evaluation_overview_ids_returns_sorted_ids(
    argilla_evaluation_repository: ArgillaEvaluationRepository,
    evaluation_overviews: Iterable[EvaluationOverview],
) -> None:
    sorted_evaluation_overview_ids = sorted(
        [overview.id for overview in evaluation_overviews]
    )
    for evaluation_overview in evaluation_overviews:
        argilla_evaluation_repository.store_evaluation_overview(evaluation_overview)

    evaluation_overview_ids = argilla_evaluation_repository.evaluation_overview_ids()

    assert evaluation_overview_ids == sorted_evaluation_overview_ids


def test_example_evaluations_returns_sorted_example_evaluations(
    argilla_evaluation_repository: ArgillaEvaluationRepository,
    argilla_evaluation_repository_with_example_evaluations: Tuple[
        str,
        ArgillaEvaluationRepository,
        list[ExampleEvaluation[DummyEvaluation]],
        list[ExampleEvaluation[DummyEvaluation]],
    ],
) -> None:
    (
        dataset_id,
        argilla_evaluation_repository,
        successful_evaluation_examples,
        failed_evaluation_examples,
    ) = argilla_evaluation_repository_with_example_evaluations
    all_sorted_evaluation_examples = sorted(
        chain(successful_evaluation_examples, failed_evaluation_examples),
        key=lambda example: example.example_id,
    )

    example_evaluations = argilla_evaluation_repository.example_evaluations(
        dataset_id, ArgillaEvaluation
    )

    assert len(example_evaluations) == len(all_sorted_evaluation_examples)
    for i, example_evaluation in enumerate(example_evaluations):
        assert (
            example_evaluation.example_id
            == all_sorted_evaluation_examples[i].example_id
        )
        assert (
            example_evaluation.evaluation_id
            == all_sorted_evaluation_examples[i].evaluation_id
        )


def test_successful_example_evaluations_returns_sorted_successful_example_evaluations(
    argilla_evaluation_repository: ArgillaEvaluationRepository,
    argilla_evaluation_repository_with_example_evaluations: Tuple[
        str,
        ArgillaEvaluationRepository,
        list[ExampleEvaluation[DummyEvaluation]],
        list[ExampleEvaluation[DummyEvaluation]],
    ],
) -> None:
    (
        dataset_id,
        argilla_evaluation_repository,
        successful_evaluation_examples,
        failed_evaluation_examples,
    ) = argilla_evaluation_repository_with_example_evaluations
    sorted_successful_evaluation_examples = sorted(
        successful_evaluation_examples, key=lambda example: example.example_id
    )

    example_evaluations = argilla_evaluation_repository.successful_example_evaluations(
        dataset_id, ArgillaEvaluation
    )

    assert len(example_evaluations) == len(sorted_successful_evaluation_examples)
    for i, example_evaluation in enumerate(example_evaluations):
        assert (
            example_evaluation.example_id
            == sorted_successful_evaluation_examples[i].example_id
        )
        assert (
            example_evaluation.evaluation_id
            == sorted_successful_evaluation_examples[i].evaluation_id
        )


def test_failed_example_evaluations_returns_sorted_failed_example_evaluations(
    argilla_evaluation_repository: ArgillaEvaluationRepository,
    argilla_evaluation_repository_with_example_evaluations: Tuple[
        str,
        ArgillaEvaluationRepository,
        list[ExampleEvaluation[DummyEvaluation]],
        list[ExampleEvaluation[DummyEvaluation]],
    ],
) -> None:
    (
        dataset_id,
        argilla_evaluation_repository,
        successful_evaluation_examples,
        failed_evaluation_examples,
    ) = argilla_evaluation_repository_with_example_evaluations
    sorted_failed_evaluation_examples = sorted(
        failed_evaluation_examples, key=lambda example: example.example_id
    )

    example_evaluations = argilla_evaluation_repository.failed_example_evaluations(
        dataset_id, ArgillaEvaluation
    )

    assert len(example_evaluations) == len(sorted_failed_evaluation_examples)
    for i, example_evaluation in enumerate(example_evaluations):
        assert (
            example_evaluation.example_id
            == sorted_failed_evaluation_examples[i].example_id
        )
        assert (
            example_evaluation.evaluation_id
            == sorted_failed_evaluation_examples[i].evaluation_id
        )
