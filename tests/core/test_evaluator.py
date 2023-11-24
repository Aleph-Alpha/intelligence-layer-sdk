from datetime import datetime
from typing import Iterable, Literal, Sequence

from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.core import (
    EvaluationException,
    Evaluator,
    Example,
    InMemoryEvaluationRepository,
    NoOpTracer,
    SequenceDataset,
    TaskSpanTrace,
    Tracer,
)
from intelligence_layer.core.evaluator import LogTrace, SpanTrace, _to_trace_entry
from intelligence_layer.core.task import Task
from intelligence_layer.core.tracer import InMemorySpan, InMemoryTaskSpan, LogEntry

DummyTaskInput = Literal["success", "fail in task", "fail in eval"]
DummyTaskOutput = DummyTaskInput


class DummyEvaluation(BaseModel):
    result: str


class AggregatedDummyEvaluation(BaseModel):
    results: Sequence[DummyEvaluation]


class DummyEvaluator(
    Evaluator[
        DummyTaskInput,
        DummyTaskOutput,
        None,
        DummyEvaluation,
        AggregatedDummyEvaluation,
    ]
):
    def do_evaluate(
        self, input: DummyTaskInput, output: DummyTaskOutput, expected_output: None
    ) -> DummyEvaluation:
        if output == "fail in eval":
            raise RuntimeError(output)
        return DummyEvaluation(result="pass")

    def aggregate(
        self, evaluations: Iterable[DummyEvaluation]
    ) -> AggregatedDummyEvaluation:
        return AggregatedDummyEvaluation(results=list(evaluations))


class DummyTask(Task[DummyTaskInput, DummyTaskOutput]):
    def do_run(self, input: DummyTaskInput, tracer: Tracer) -> DummyTaskOutput:
        if input == "fail in task":
            raise RuntimeError(input)
        return input


@fixture
def evaluation_repository() -> InMemoryEvaluationRepository:
    return InMemoryEvaluationRepository()


@fixture
def dummy_evaluator(
    evaluation_repository: InMemoryEvaluationRepository,
) -> DummyEvaluator:
    return DummyEvaluator(DummyTask(), evaluation_repository)


def test_evaluate_dataset_returns_generic_statistics(
    dummy_evaluator: DummyEvaluator,
) -> None:
    test_start = datetime.utcnow()
    examples: Sequence[Example[DummyTaskInput, None]] = [
        Example(input="success", expected_output=None),
        Example(input="fail in task", expected_output=None),
        Example(input="fail in eval", expected_output=None),
    ]

    dataset: SequenceDataset[DummyTaskInput, None] = SequenceDataset(
        name="test",
        examples=examples,
    )

    evaluation_run_overview = dummy_evaluator.evaluate_dataset(dataset, NoOpTracer())

    assert evaluation_run_overview.dataset_name == dataset.name
    assert test_start <= evaluation_run_overview.start <= evaluation_run_overview.end
    assert evaluation_run_overview.failed_evaluation_count == 2
    assert evaluation_run_overview.successful_evaluation_count == 1


def test_evaluate_dataset_stores_example_results(
    dummy_evaluator: DummyEvaluator,
) -> None:
    evaluation_repository = dummy_evaluator.repository
    examples: Sequence[Example[DummyTaskInput, None]] = [
        Example(input="success", expected_output=None),
        Example(input="fail in task", expected_output=None),
        Example(input="fail in eval", expected_output=None),
    ]

    dataset: SequenceDataset[DummyTaskInput, None] = SequenceDataset(
        name="test",
        examples=examples,
    )

    evaluation_run_overview = dummy_evaluator.evaluate_dataset(dataset, NoOpTracer())
    success_result = evaluation_repository.evaluation_example_result(
        evaluation_run_overview.id, examples[0].id, DummyEvaluation
    )
    failure_result_task = evaluation_repository.evaluation_example_result(
        evaluation_run_overview.id, examples[1].id, DummyEvaluation
    )
    failure_result_eval = evaluation_repository.evaluation_example_result(
        evaluation_run_overview.id, examples[2].id, DummyEvaluation
    )

    assert success_result and isinstance(success_result.result, DummyEvaluation)
    assert failure_result_task and isinstance(
        failure_result_task.result, EvaluationException
    )
    assert failure_result_eval and isinstance(
        failure_result_eval.result, EvaluationException
    )
    assert success_result.trace.input == "success"
    assert failure_result_task.trace.input == "fail in task"
    assert failure_result_eval.trace.input == "fail in eval"


def test_evaluate_dataset_stores_aggregated_results(
    dummy_evaluator: DummyEvaluator,
) -> None:
    evaluation_repository = dummy_evaluator.repository

    dataset: SequenceDataset[DummyTaskInput, None] = SequenceDataset(
        name="test",
        examples=[],
    )

    evaluation_run_overview = dummy_evaluator.evaluate_dataset(dataset, NoOpTracer())
    loaded_evaluation_run_overview = evaluation_repository.evaluation_run_overview(
        evaluation_run_overview.id, AggregatedDummyEvaluation
    )

    assert evaluation_run_overview == loaded_evaluation_run_overview


def test_to_trace_entry() -> None:
    now = datetime.utcnow()
    entry = _to_trace_entry(
        InMemoryTaskSpan(
            name="ignored",
            input="input",
            output="output",
            start_timestamp=now,
            end_timestamp=now,
            entries=[
                LogEntry(message="message", value="value"),
                InMemorySpan(name="ignored", start_timestamp=now, end_timestamp=now),
            ],
        )
    )

    assert entry == TaskSpanTrace(
        input="input",
        output="output",
        start=now,
        end=now,
        traces=[
            LogTrace(message="message", value="value"),
            SpanTrace(traces=[], start=now, end=now),
        ],
    )


def test_deserialize_task_trace() -> None:
    trace = TaskSpanTrace(
        start=datetime.utcnow(),
        end=datetime.utcnow(),
        traces=[],
        input=[{"a": "b"}],
        output=["c"],
    )
    assert trace.model_validate_json(trace.model_dump_json()) == trace
