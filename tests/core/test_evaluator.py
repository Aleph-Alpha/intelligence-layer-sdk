from datetime import datetime
from typing import Iterable, Optional

from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.core import (
    Dataset,
    EvaluationException,
    Evaluator,
    Example,
    InMemoryEvaluationRepository,
    NoOpTracer,
    SequenceDataset,
    TaskTrace,
    Tracer,
)
from intelligence_layer.core.evaluator import SpanTrace, TraceLog, to_trace_entry
from intelligence_layer.core.task import Task
from intelligence_layer.core.tracer import InMemorySpan, InMemoryTaskSpan, LogEntry


class DummyEvaluation(BaseModel):
    result: str


class DummyEvaluator(
    Evaluator[Optional[str], Optional[str], None, DummyEvaluation, None]
):
    def do_evaluate(
        self, input: Optional[str], output: Optional[str], expected_output: None
    ) -> DummyEvaluation:
        if output is None:
            return DummyEvaluation(result="pass")
        raise RuntimeError(output)

    def aggregate(self, evaluations: Iterable[DummyEvaluation]) -> None:
        return None


class DummyTask(Task[Optional[str], Optional[str]]):
    def do_run(self, input: str | None, tracer: Tracer) -> str | None:
        return input


@fixture
def evaluation_repository() -> InMemoryEvaluationRepository:
    return InMemoryEvaluationRepository()


@fixture
def dummy_evaluator(
    evaluation_repository: InMemoryEvaluationRepository,
) -> DummyEvaluator:
    return DummyEvaluator(DummyTask(), evaluation_repository)


def test_evaluate_dataset_does_not_throw_an_exception_for_failure(
    dummy_evaluator: DummyEvaluator,
) -> None:
    dataset: Dataset[Optional[str], None] = SequenceDataset(
        name="test",
        examples=[Example(input="fail", expected_output=None)],
    )
    dummy_evaluator.evaluate_dataset(dataset, NoOpTracer())


def test_evaluate_dataset_stores_example_results(
    dummy_evaluator: DummyEvaluator,
) -> None:
    evaluation_repository = dummy_evaluator.repository
    dataset: SequenceDataset[str | None, None] = SequenceDataset(
        name="test",
        examples=[
            Example(input=None, expected_output=None),
            Example(input="fail", expected_output=None),
        ],
    )

    evaluation_run_overview = dummy_evaluator.evaluate_dataset(dataset, NoOpTracer())
    results = evaluation_repository.evaluation_run_results(
        evaluation_run_overview.id, DummyEvaluation
    )

    assert isinstance(results[0].result, DummyEvaluation)
    assert isinstance(results[1].result, EvaluationException)
    assert [isinstance(r.trace, TaskTrace) for r in results]
    assert len(results) == len(dataset.examples)


def test_to_trace_entry() -> None:
    now = datetime.utcnow()
    entry = to_trace_entry(
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

    assert entry == TaskTrace(
        input="input",
        output="output",
        start=now,
        end=now,
        traces=[
            TraceLog(message="message", value="value"),
            SpanTrace(traces=[], start=now, end=now),
        ],
    )


def test_deserialize_task_trace() -> None:
    trace = TaskTrace(
        start=datetime.utcnow(),
        end=datetime.utcnow(),
        traces=[],
        input=[{"a": "b"}],
        output=["c"],
    )
    assert trace.model_validate_json(trace.model_dump_json()) == trace
