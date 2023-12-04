from datetime import datetime
from typing import Sequence, cast

from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.core import (
    ExampleEvaluation,
    ExampleTrace,
    FailedExampleEvaluation,
    FileEvaluationRepository,
    InMemoryEvaluationRepository,
    InMemoryTaskSpan,
    TaskSpanTrace,
)
from intelligence_layer.core.evaluation.domain import (
    EvaluationOverview,
    EvaluationRunOverview,
    ExampleOutput,
    RunOverview,
)
from intelligence_layer.core.tracer import CompositeTracer, InMemoryTracer


class DummyEvaluation(BaseModel):
    result: str


class DummyAggregatedEvaluation(BaseModel):
    score: float


class DummyEvaluationWithExceptionStructure(BaseModel):
    error_message: str


class DummyTaskInput(BaseModel):
    input: str


@fixture
def task_span_trace() -> TaskSpanTrace:
    now = datetime.now()
    return TaskSpanTrace(
        name="task name", traces=[], start=now, end=now, input="input", output="output"
    )


@fixture
def successful_example_result() -> ExampleEvaluation[DummyEvaluation]:
    return ExampleEvaluation(
        example_id="example_id",
        result=DummyEvaluation(result="result"),
    )


@fixture
def example_trace(
    task_span_trace: TaskSpanTrace,
) -> ExampleTrace:
    return ExampleTrace(
        example_id="example_id",
        trace=task_span_trace,
    )


@fixture
def failed_example_result() -> ExampleEvaluation[DummyEvaluation]:
    return ExampleEvaluation(
        example_id="other",
        result=FailedExampleEvaluation(error_message="error"),
    )


def test_can_store_example_evaluation_traces_in_file(
    file_evaluation_repository: FileEvaluationRepository,
) -> None:
    run_id = "run_id"
    example_id = "example_id"
    now = datetime.now()

    tracer = file_evaluation_repository.example_tracer(run_id, example_id)
    expected = InMemoryTracer()
    CompositeTracer([tracer, expected]).task_span(
        "task", DummyTaskInput(input="input"), now
    )

    assert file_evaluation_repository.example_trace(run_id, example_id) == ExampleTrace(
        example_id=example_id,
        trace=TaskSpanTrace.from_task_span(cast(InMemoryTaskSpan, expected.entries[0])),
    )


def test_can_store_example_results_in_file(
    file_evaluation_repository: FileEvaluationRepository,
    successful_example_result: ExampleEvaluation[DummyEvaluation],
) -> None:
    run_id = "id"

    file_evaluation_repository.store_example_evaluation(
        run_id, successful_example_result
    )

    assert (
        file_evaluation_repository.example_evaluation(
            run_id, successful_example_result.example_id, DummyEvaluation
        )
        == successful_example_result
    )


def test_storing_exception_with_same_structure_as_type_still_deserializes_exception(
    file_evaluation_repository: FileEvaluationRepository,
) -> None:
    exception: ExampleEvaluation[DummyEvaluation] = ExampleEvaluation(
        example_id="id",
        result=FailedExampleEvaluation(error_message="error"),
    )
    run_id = "id"

    file_evaluation_repository.store_example_evaluation(run_id, exception)

    assert (
        file_evaluation_repository.example_evaluation(
            run_id, exception.example_id, DummyEvaluationWithExceptionStructure
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
) -> None:
    run_id = "id"
    results: Sequence[ExampleEvaluation[DummyEvaluation]] = [
        successful_example_result,
        failed_example_result,
    ]
    for result in results:
        file_evaluation_repository.store_example_evaluation(run_id, result)

    run_results = file_evaluation_repository.example_evaluations(
        run_id, DummyEvaluation
    )

    assert sorted(results, key=lambda i: i.example_id) == sorted(
        run_results, key=lambda i: i.example_id
    )


def test_file_repository_can_fetch_failed_examples_from_evaluation_run(
    file_evaluation_repository: FileEvaluationRepository,
    successful_example_result: ExampleEvaluation[DummyEvaluation],
    failed_example_result: ExampleEvaluation[DummyEvaluation],
) -> None:
    run_id = "id"
    results: Sequence[ExampleEvaluation[DummyEvaluation]] = [
        successful_example_result,
        failed_example_result,
    ]
    for result in results:
        file_evaluation_repository.store_example_evaluation(run_id, result)

    run_results = file_evaluation_repository.failed_example_evaluations(
        run_id, DummyEvaluation
    )

    assert run_results == [failed_example_result]


def test_in_memory_repository_can_fetch_failed_examples_from_evaluation_run(
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    successful_example_result: ExampleEvaluation[DummyEvaluation],
    failed_example_result: ExampleEvaluation[DummyEvaluation],
) -> None:
    run_id = "id"
    results: Sequence[ExampleEvaluation[DummyEvaluation]] = [
        successful_example_result,
        failed_example_result,
    ]
    for result in results:
        in_memory_evaluation_repository.store_example_evaluation(run_id, result)

    run_results = in_memory_evaluation_repository.failed_example_evaluations(
        run_id, DummyEvaluation
    )

    assert run_results == [failed_example_result]


def test_file_repository_returns_empty_sequence_for_non_existing_run_id(
    file_evaluation_repository: FileEvaluationRepository,
) -> None:
    run_id = "does-not-exist"

    assert file_evaluation_repository.example_evaluations(run_id, DummyEvaluation) == []


def test_file_repository_stores_overview(
    file_evaluation_repository: FileEvaluationRepository,
) -> None:
    now = datetime.now()
    overview = EvaluationRunOverview(
        evaluation_overview=EvaluationOverview(
            id="eval-id",
            run_overview=RunOverview(
                dataset_name="dataset",
                id="run-id",
                start=now,
                end=now,
                failed_example_count=0,
                successful_example_count=0,
            ),
            failed_evaluation_count=3,
            successful_evaluation_count=5,
            start=now,
            end=now,
        ),
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


def test_file_repository_run_id_returns_run_ids(
    file_evaluation_repository: FileEvaluationRepository,
) -> None:
    run_id = "id"

    file_evaluation_repository.store_example_output(
        run_id, ExampleOutput(example_id="exmaple_id", output=None)
    )

    assert file_evaluation_repository.run_ids() == [run_id]
