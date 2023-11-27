from datetime import datetime

from intelligence_layer.core import (
    InMemorySpan,
    InMemoryTaskSpan,
    LogEntry,
    LogTrace,
    SpanTrace,
    TaskSpanTrace,
)
from intelligence_layer.core.evaluation.domain import _to_trace_entry


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
