from datetime import datetime
from typing import Sequence

from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.evaluation import (
    ExampleEvaluation,
    ExampleTrace,
    FailedExampleEvaluation,
    FileEvaluationRepository,
    InMemoryEvaluationRepository,
    TaskSpanTrace,
)
from intelligence_layer.evaluation.domain import EvaluationOverview
from tests.evaluation.conftest import DummyEvaluation


class DummyEvaluationWithExceptionStructure(BaseModel):
    error_message: str


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
        run_id="some_eval_id",
        example_id="example_id",
        trace=task_span_trace,
    )


def test_can_store_example_results_in_file(
    file_evaluation_repository: FileEvaluationRepository,
    successful_example_result: ExampleEvaluation[DummyEvaluation],
    eval_id: str,
) -> None:
    file_evaluation_repository.store_example_evaluation(successful_example_result)

    assert (
        file_evaluation_repository.example_evaluation(
            eval_id,
            successful_example_result.example_id,
            DummyEvaluation,
        )
        == successful_example_result
    )


def test_storing_exception_with_same_structure_as_type_still_deserializes_exception(
    file_evaluation_repository: FileEvaluationRepository,
) -> None:
    eval_id = "eval_id"
    exception: ExampleEvaluation[DummyEvaluation] = ExampleEvaluation(
        eval_id=eval_id,
        example_id="example_id",
        result=FailedExampleEvaluation(error_message="error"),
    )

    file_evaluation_repository.store_example_evaluation(exception)

    assert (
        file_evaluation_repository.example_evaluation(
            eval_id, exception.example_id, DummyEvaluationWithExceptionStructure
        )
        == exception
    )


def test_file_repository_returns_none_in_case_example_result_does_not_exist(
    file_evaluation_repository: FileEvaluationRepository,
) -> None:
    assert (
        file_evaluation_repository.example_evaluation("id", "id", DummyEvaluation)
        is None
    )


def test_file_repository_can_fetch_full_evaluation_runs(
    file_evaluation_repository: FileEvaluationRepository,
    successful_example_result: ExampleEvaluation[DummyEvaluation],
    failed_example_result: ExampleEvaluation[DummyEvaluation],
    eval_id: str,
) -> None:
    results: Sequence[ExampleEvaluation[DummyEvaluation]] = [
        successful_example_result,
        failed_example_result,
    ]
    for result in results:
        file_evaluation_repository.store_example_evaluation(result)

    run_results = file_evaluation_repository.example_evaluations(
        eval_id, DummyEvaluation
    )

    assert sorted(results, key=lambda i: i.example_id) == sorted(
        run_results, key=lambda i: i.example_id
    )


def test_file_repository_can_fetch_failed_examples_from_evaluation_run(
    file_evaluation_repository: FileEvaluationRepository,
    successful_example_result: ExampleEvaluation[DummyEvaluation],
    failed_example_result: ExampleEvaluation[DummyEvaluation],
    eval_id: str,
) -> None:
    results: Sequence[ExampleEvaluation[DummyEvaluation]] = [
        successful_example_result,
        failed_example_result,
    ]
    for result in results:
        file_evaluation_repository.store_example_evaluation(result)

    run_results = file_evaluation_repository.failed_example_evaluations(
        eval_id, DummyEvaluation
    )

    assert run_results == [failed_example_result]


def test_in_memory_repository_can_fetch_failed_examples_from_evaluation_run(
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    successful_example_result: ExampleEvaluation[DummyEvaluation],
    failed_example_result: ExampleEvaluation[DummyEvaluation],
    eval_id: str,
) -> None:
    results: Sequence[ExampleEvaluation[DummyEvaluation]] = [
        successful_example_result,
        failed_example_result,
    ]
    for result in results:
        in_memory_evaluation_repository.store_example_evaluation(result)

    run_results = in_memory_evaluation_repository.failed_example_evaluations(
        eval_id, DummyEvaluation
    )

    assert run_results == [failed_example_result]


def test_file_repository_returns_empty_sequence_for_non_existing_run_id(
    file_evaluation_repository: FileEvaluationRepository,
) -> None:
    run_id = "does-not-exist"

    assert file_evaluation_repository.example_evaluations(run_id, DummyEvaluation) == []


def test_file_repository_stores_overview(
    file_evaluation_repository: FileEvaluationRepository,
    evaluation_overview: EvaluationOverview,
) -> None:
    file_evaluation_repository.store_evaluation_overview(evaluation_overview)
    assert (
        file_evaluation_repository.evaluation_overview(evaluation_overview.id)
        == evaluation_overview
    )


def test_file_repository_returns_none_for_nonexisting_overview(
    file_evaluation_repository: FileEvaluationRepository,
) -> None:
    assert file_evaluation_repository.evaluation_overview("does-not-exist") is None
