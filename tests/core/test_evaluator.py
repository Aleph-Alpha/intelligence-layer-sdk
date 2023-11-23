from datetime import datetime
from typing import Iterable, Optional, Sequence

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
    TaskSpanTrace,
    Tracer,
)
from intelligence_layer.core.evaluator import LogTrace, SpanTrace, to_trace_entry
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
        list(evaluations)


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
    examples: Sequence[Example[str | None, None]] = [
        Example(input=None, expected_output=None),
        Example(input="fail", expected_output=None),
    ]

    dataset: SequenceDataset[str | None, None] = SequenceDataset(
        name="test",
        examples=examples,
    )

    evaluation_run_overview = dummy_evaluator.evaluate_dataset(dataset, NoOpTracer())
    success_result = evaluation_repository.evaluation_example_result(
        evaluation_run_overview.id, examples[0].id, DummyEvaluation
    )
    failure_result = evaluation_repository.evaluation_example_result(
        evaluation_run_overview.id, examples[1].id, DummyEvaluation
    )

    assert success_result and isinstance(success_result.result, DummyEvaluation)
    assert failure_result and isinstance(failure_result.result, EvaluationException)
    assert success_result.trace.input is None
    assert failure_result.trace.input == "fail"


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


# TODO

# - check ci problem
# - add documentation to evaluator
# - store aggregation result
# - file bases repo
# - refactor _rich_render_ (reuse in tracer and evaluator?)
# - introduce MappingTask (to remove redundancy in ClassifyEvaluators)
