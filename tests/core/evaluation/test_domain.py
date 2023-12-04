from datetime import datetime

from pydantic import BaseModel
from pytest import raises

from intelligence_layer.core import (
    InMemorySpan,
    InMemoryTaskSpan,
    LogEntry,
    LogTrace,
    SpanTrace,
    TaskSpanTrace,
)
from intelligence_layer.core.evaluation.domain import (
    EvaluationFailed,
    EvaluationOverview,
    EvaluationRunOverview,
    RunOverview,
    _to_trace_entry,
)


def test_to_trace_entry() -> None:
    now = datetime.utcnow()
    entry = _to_trace_entry(
        InMemoryTaskSpan(
            name="task",
            input="input",
            output="output",
            start_timestamp=now,
            end_timestamp=now,
            entries=[
                LogEntry(message="message", value="value"),
                InMemorySpan(name="span", start_timestamp=now, end_timestamp=now),
            ],
        )
    )

    assert entry == TaskSpanTrace(
        name="task",
        input="input",
        output="output",
        start=now,
        end=now,
        traces=[
            LogTrace(message="message", value="value"),
            SpanTrace(name="span", traces=[], start=now, end=now),
        ],
    )


def test_deserialize_task_trace() -> None:
    trace = TaskSpanTrace(
        name="task",
        start=datetime.utcnow(),
        end=datetime.utcnow(),
        traces=[],
        input=[{"a": "b"}],
        output=["c"],
    )
    assert trace.model_validate_json(trace.model_dump_json()) == trace


class StatisticsDummy(BaseModel):
    result: str


def test_raise_on_exception_for_evaluation_run_overview() -> None:
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
            failed_evaluation_count=1,
            successful_evaluation_count=0,
            start=now,
            end=now,
        ),
        statistics=StatisticsDummy(result="result"),
    )

    with raises(EvaluationFailed):
        overview.raise_on_evaluation_failure()
