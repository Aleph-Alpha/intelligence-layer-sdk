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
from intelligence_layer.core.evaluation.domain import EvaluationOverview, ExampleOutput
from intelligence_layer.core.evaluation.evaluator import EvaluationRepository
from intelligence_layer.core.tracer import CompositeTracer, InMemoryTracer
from tests.conftest import DummyStringInput
from tests.core.evaluation.conftest import DummyAggregatedEvaluation, DummyEvaluation


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
        example_id="example_id",
        trace=task_span_trace,
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
        "task", DummyStringInput(input="input"), now
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
    evaluation_run_overview: EvaluationOverview[DummyAggregatedEvaluation],
) -> None:
    file_evaluation_repository.store_evaluation_overview(evaluation_run_overview)
    assert (
        file_evaluation_repository.evaluation_overview(
            evaluation_run_overview.id, EvaluationOverview[DummyAggregatedEvaluation]
        )
        == evaluation_run_overview
    )


def test_file_repository_returns_none_for_nonexisting_overview(
    file_evaluation_repository: FileEvaluationRepository,
) -> None:
    assert (
        file_evaluation_repository.evaluation_overview(
            "does-not-exist", EvaluationOverview[DummyAggregatedEvaluation]
        )
        is None
    )


def test_file_repository_run_id_returns_run_ids(
    file_evaluation_repository: FileEvaluationRepository,
) -> None:
    run_id = "id"

    file_evaluation_repository.store_example_output(
        ExampleOutput(run_id=run_id, example_id="example_id", output=None)
    )

    assert file_evaluation_repository.run_ids() == [run_id]


def evaluation_repository_returns_examples_in_same_order_for_two_runs(
    evaluation_repository: EvaluationRepository,
) -> None:
    run_id_1 = "id_1"
    run_id_2 = "id_2"
    num_examples = 20

    for example_id in range(num_examples):
        evaluation_repository.store_example_output(
            ExampleOutput(run_id=run_id_1, example_id=str(example_id), output=None),
        )

    for example_id in reversed(range(num_examples)):
        evaluation_repository.store_example_output(
            ExampleOutput(run_id=run_id_2, example_id=str(example_id), output=None),
        )

    assert list(
        (output.example_id, output.output)
        for output in evaluation_repository.example_outputs(run_id_1, type(None))
    ) == list(
        (output.example_id, output.output)
        for output in evaluation_repository.example_outputs(run_id_2, type(None))
    )


def test_in_memory_evaluation_repository_returns_examples_in_same_order_for_two_runs(
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
) -> None:
    evaluation_repository_returns_examples_in_same_order_for_two_runs(
        in_memory_evaluation_repository
    )


def test_file_evaluation_repository_returns_examples_in_same_order_for_two_runs(
    file_evaluation_repository: FileEvaluationRepository,
) -> None:
    evaluation_repository_returns_examples_in_same_order_for_two_runs(
        file_evaluation_repository
    )
