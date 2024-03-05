from datetime import datetime
from typing import cast

from intelligence_layer.core import CompositeTracer, InMemoryTaskSpan, InMemoryTracer
from intelligence_layer.evaluation import ExampleTrace, TaskSpanTrace
from intelligence_layer.evaluation.data_storage.run_repository import (
    FileRunRepository,
    RunRepository,
)
from intelligence_layer.evaluation.domain import ExampleOutput
from tests.conftest import DummyStringInput


def test_can_store_example_evaluation_traces_in_file(
    file_run_repository: FileRunRepository,
) -> None:
    run_id = "run_id"
    example_id = "example_id"
    now = datetime.now()

    tracer = file_run_repository.example_tracer(run_id, example_id)
    expected = InMemoryTracer()
    CompositeTracer([tracer, expected]).task_span(
        "task", DummyStringInput(input="input"), now
    )

    assert file_run_repository.example_trace(run_id, example_id) == ExampleTrace(
        run_id=run_id,
        example_id=example_id,
        trace=TaskSpanTrace.from_task_span(cast(InMemoryTaskSpan, expected.entries[0])),
    )


def test_file_repository_run_id_returns_run_ids(
    file_run_repository: FileRunRepository,
) -> None:
    run_id = "id"

    file_run_repository.store_example_output(
        ExampleOutput(run_id=run_id, example_id="example_id", output=None)
    )

    assert file_run_repository.example_output_ids() == [run_id]


# def test_in_memory_evaluation_repository_returns_examples_in_same_order_for_two_runs(
#     in_memory_evaluation_repository: InMemoryEvaluationRepository,
# ) -> None:
#     evaluation_repository_returns_examples_in_same_order_for_two_runs(
#         in_memory_evaluation_repository
#     )


def test_file_evaluation_repository_returns_examples_in_same_order_for_two_runs(
    file_run_repository: FileRunRepository,
) -> None:
    evaluation_repository_returns_examples_in_same_order_for_two_runs(
        file_run_repository
    )


def evaluation_repository_returns_examples_in_same_order_for_two_runs(
    run_repository: RunRepository,
) -> None:
    run_id_1 = "id_1"
    run_id_2 = "id_2"
    num_examples = 20

    for example_id in range(num_examples):
        run_repository.store_example_output(
            ExampleOutput(run_id=run_id_1, example_id=str(example_id), output=None),
        )

    for example_id in reversed(range(num_examples)):
        run_repository.store_example_output(
            ExampleOutput(run_id=run_id_2, example_id=str(example_id), output=None),
        )

    assert list(
        (output.example_id, output.output)
        for output in run_repository.example_outputs(run_id_1, type(None))
    ) == list(
        (output.example_id, output.output)
        for output in run_repository.example_outputs(run_id_2, type(None))
    )
