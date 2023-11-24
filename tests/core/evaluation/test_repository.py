from datetime import datetime
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.core import (
    EvaluationException,
    ExampleResult,
    FileEvaluationRepository,
    TaskSpanTrace,
)
from intelligence_layer.core.evaluation.domain import EvaluationRunOverview


class DummyEvaluation(BaseModel):
    result: str


class DummyAggregatedEvaluation(BaseModel):
    score: float


class DummyEvaluationWithExceptionStructure(BaseModel):
    error_message: str


@fixture
def file_evaluation_repository(tmp_path: Path) -> FileEvaluationRepository:
    return FileEvaluationRepository(tmp_path)


@fixture
def task_span_trace() -> TaskSpanTrace:
    now = datetime.now()
    return TaskSpanTrace(traces=[], start=now, end=now, input="input", output="output")


@fixture
def successful_example_result(
    task_span_trace: TaskSpanTrace,
) -> ExampleResult[DummyEvaluation]:
    return ExampleResult(
        example_id="example_id",
        result=DummyEvaluation(result="result"),
        trace=task_span_trace,
    )


def test_can_store_traces_in_file(
    file_evaluation_repository: FileEvaluationRepository,
    successful_example_result: ExampleResult[DummyEvaluation],
) -> None:
    run_id = "id"

    file_evaluation_repository.store_example_result(run_id, successful_example_result)

    assert (
        file_evaluation_repository.evaluation_example_result(
            run_id, successful_example_result.example_id, DummyEvaluation
        )
        == successful_example_result
    )


def test_storing_exception_with_same_structure_as_type_still_deserializes_exception(
    file_evaluation_repository: FileEvaluationRepository, task_span_trace: TaskSpanTrace
) -> None:
    exception: ExampleResult[DummyEvaluation] = ExampleResult(
        example_id="id",
        result=EvaluationException(error_message="error"),
        trace=task_span_trace,
    )
    run_id = "id"

    file_evaluation_repository.store_example_result(run_id, exception)

    assert (
        file_evaluation_repository.evaluation_example_result(
            run_id, exception.example_id, DummyEvaluationWithExceptionStructure
        )
        == exception
    )


def test_file_repository_returns_none_in_case_example_result_does_not_exist(
    file_evaluation_repository: FileEvaluationRepository,
) -> None:
    assert (
        file_evaluation_repository.evaluation_example_result(
            "id", "id", DummyEvaluation
        )
        is None
    )


def test_file_repository_can_fetch_full_evaluation_runs(
    file_evaluation_repository: FileEvaluationRepository,
    successful_example_result: ExampleResult[DummyEvaluation],
    task_span_trace: TaskSpanTrace,
) -> None:
    run_id = "id"
    results: Sequence[ExampleResult[DummyEvaluation]] = [
        successful_example_result,
        ExampleResult(
            example_id="other",
            result=EvaluationException(error_message="error"),
            trace=task_span_trace,
        ),
    ]
    for result in results:
        file_evaluation_repository.store_example_result(run_id, result)

    run_results = file_evaluation_repository.evaluation_run_results(
        run_id, DummyEvaluation
    )

    assert sorted(results, key=lambda i: i.example_id) == sorted(
        run_results, key=lambda i: i.example_id
    )


def test_file_repository_returns_empty_sequence_for_non_existing_run_id(
    file_evaluation_repository: FileEvaluationRepository,
) -> None:
    run_id = "does-not-exist"

    assert (
        file_evaluation_repository.evaluation_run_results(run_id, DummyEvaluation) == []
    )


def test_file_repository_stores_overview(
    file_evaluation_repository: FileEvaluationRepository,
) -> None:
    now = datetime.now()
    overview = EvaluationRunOverview(
        id="id",
        dataset_name="name",
        failed_evaluation_count=3,
        successful_evaluation_count=5,
        start=now,
        end=now,
        statistics=DummyAggregatedEvaluation(score=0.5),
    )

    file_evaluation_repository.store_evaluation_run_overview(overview)

    assert (
        file_evaluation_repository.evaluation_run_overview(
            overview.id, DummyAggregatedEvaluation
        )
        == overview
    )


def test_file_repository_returns_none_for_nonexisting_overview(
    file_evaluation_repository: FileEvaluationRepository,
) -> None:
    assert (
        file_evaluation_repository.evaluation_run_overview(
            "does-not-exist", DummyAggregatedEvaluation
        )
        is None
    )
