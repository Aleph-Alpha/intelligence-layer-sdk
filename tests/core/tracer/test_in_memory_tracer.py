import contextlib
import os
from typing import Iterator

import pytest

from intelligence_layer.core import (
    InMemorySpan,
    InMemoryTaskSpan,
    InMemoryTracer,
    LogEntry,
    Task,
    utc_now,
)
from intelligence_layer.core.tracer.tracer import ErrorValue


def test_trace_id_exists_for_all_children_of_task_span() -> None:
    tracer = InMemoryTracer()
    parent_span = tracer.task_span("child", "input")
    parent_span.span("child2")

    assert isinstance(tracer.entries[0], InMemoryTaskSpan)
    assert isinstance(tracer.entries[0].entries[0], InMemorySpan)
    assert (
        tracer.entries[0].entries[0].context.trace_id
        == tracer.entries[0].context.trace_id
    )

    parent_span.task_span("child3", "input")
    assert isinstance(tracer.entries[0].entries[1], InMemoryTaskSpan)
    assert (
        tracer.entries[0].entries[1].context.trace_id
        == tracer.entries[0].context.trace_id
    )


def test_trace_id_exists_for_all_children_of_span() -> None:
    tracer = InMemoryTracer()
    parent_span = tracer.span("child")
    parent_span.span("child2")

    assert isinstance(tracer.entries[0], InMemorySpan)
    assert isinstance(tracer.entries[0].entries[0], InMemorySpan)
    assert (
        tracer.entries[0].entries[0].context.trace_id
        == tracer.entries[0].context.trace_id
    )

    parent_span.task_span("child3", "input")
    assert isinstance(tracer.entries[0].entries[1], InMemorySpan)
    assert (
        tracer.entries[0].entries[1].context.trace_id
        == tracer.entries[0].context.trace_id
    )


def test_can_add_child_tracer() -> None:
    tracer = InMemoryTracer()
    tracer.span("child")

    assert len(tracer.entries) == 1

    log = tracer.entries[0]
    assert isinstance(log, InMemoryTracer)
    assert log.name == "child"
    assert len(log.entries) == 0


def test_can_add_parent_and_child_entries() -> None:
    parent = InMemoryTracer()
    with parent.span("child") as child:
        child.log("Two", 2)

    assert isinstance(parent.entries[0], InMemoryTracer)
    assert isinstance(parent.entries[0].entries[0], LogEntry)


def test_task_logs_error_value() -> None:
    tracer = InMemoryTracer()

    with pytest.raises(ValueError):
        with tracer.span("failing task"):
            raise ValueError("my bad, sorry")

    assert isinstance(tracer.entries[0], InMemorySpan)
    assert isinstance(tracer.entries[0].entries[0], LogEntry)
    error = tracer.entries[0].entries[0].value
    assert isinstance(error, ErrorValue)
    assert error.message == "my bad, sorry"
    assert error.error_type == "ValueError"
    assert error.stack_trace.startswith("Traceback")


def test_task_span_records_error_value() -> None:
    tracer = InMemoryTracer()

    with pytest.raises(ValueError):
        with tracer.task_span("failing task", None):
            raise ValueError("my bad, sorry")

    assert isinstance(tracer.entries[0], InMemoryTaskSpan)
    error = tracer.entries[0].output
    assert isinstance(error, ErrorValue)
    assert error.message == "my bad, sorry"
    assert error.error_type == "ValueError"
    assert error.stack_trace.startswith("Traceback")


def test_task_automatically_logs_input_and_output(
    test_task: Task[str, str],
) -> None:
    input = "input"
    tracer = InMemoryTracer()
    output = test_task.run(input=input, tracer=tracer)

    assert len(tracer.entries) == 1
    task_span = tracer.entries[0]
    assert isinstance(task_span, InMemoryTaskSpan)
    assert task_span.name == type(test_task).__name__
    assert task_span.input == input
    assert task_span.output == output
    assert task_span.start_timestamp and task_span.end_timestamp
    assert task_span.start_timestamp < task_span.end_timestamp


def test_tracer_can_set_custom_start_time_for_log_entry() -> None:
    tracer = InMemoryTracer()
    timestamp = utc_now()

    with tracer.span("span") as span:
        span.log("log", "message", timestamp)

    assert isinstance(tracer.entries[0], InMemorySpan)
    assert isinstance(tracer.entries[0].entries[0], LogEntry)
    assert tracer.entries[0].entries[0].timestamp == timestamp


def test_tracer_can_set_custom_start_time_for_span() -> None:
    tracer = InMemoryTracer()
    start = utc_now()

    span = tracer.span("span", start)

    assert span.start_timestamp == start


def test_span_sets_end_timestamp() -> None:
    tracer = InMemoryTracer()
    start = utc_now()

    span = tracer.span("span", start)
    span.end()

    assert span.end_timestamp and span.start_timestamp <= span.end_timestamp


def test_span_only_updates_end_timestamp_once() -> None:
    tracer = InMemoryTracer()

    span = tracer.span("span")
    end = utc_now()
    span.end(end)
    span.end()

    assert span.end_timestamp == end


# take from and modified: https://stackoverflow.com/questions/2059482/temporarily-modify-the-current-processs-environment
@contextlib.contextmanager
def set_env(name: str, value: str | None) -> Iterator[None]:
    old_environ = dict(os.environ)
    if value is None:
        if os.getenv(name, None) is not None:
            os.environ.pop(name)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


def test_in_memory_tracer_trace_viewer_doesnt_crash_if_it_cant_reach_document_index() -> (
    None
):
    # note that this test sets the environment variable, which might
    # become a problem with multi-worker tests
    ENV_VARIABLE_NAME = "TRACE_VIEWER_URL"
    # ensure that the code works even with the variable is not set
    with set_env(ENV_VARIABLE_NAME, None):
        expected = InMemoryTracer()
        expected._ipython_display_()
