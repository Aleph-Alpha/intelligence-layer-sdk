from intelligence_layer.core import utc_now
from intelligence_layer.core.tracer.in_memory_tracer import InMemoryTracer
from intelligence_layer.evaluation import LogTrace, SpanTrace, TaskSpanTrace
from intelligence_layer.evaluation.run.trace import _to_trace_entry


def test_to_trace_entry() -> None:
    now = utc_now()
    span = InMemoryTracer().task_span("task", timestamp=now, input="input")
    span.span("span", now).end(now)
    span.log(message="message", value="value", timestamp=now)
    span.record_output("output")
    span.end(now)

    entry = _to_trace_entry(span)

    assert entry == TaskSpanTrace(
        name="task",
        input="input",
        output="output",
        start=now,
        end=now,
        traces=[
            SpanTrace(name="span", traces=[], start=now, end=now),
            LogTrace(message="message", value="value"),
        ],
    )


def test_deserialize_task_trace() -> None:
    trace = TaskSpanTrace(
        name="task",
        start=utc_now(),
        end=utc_now(),
        traces=[
            SpanTrace(name="span", traces=[], start=utc_now(), end=utc_now()),
            LogTrace(message="message", value="value"),
        ],
        input=[{"a": "b"}],
        output=["c"],
    )
    assert trace.model_validate_json(trace.model_dump_json()) == trace
