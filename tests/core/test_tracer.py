import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterator, Optional
from unittest.mock import Mock

import pytest
import requests
from aleph_alpha_client import Prompt
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pytest import fixture

from intelligence_layer.core import (
    CompleteInput,
    CompleteOutput,
    CompositeTracer,
    FileTracer,
    InMemorySpan,
    InMemoryTaskSpan,
    InMemoryTracer,
    LogEntry,
    LuminousControlModel,
    OpenTelemetryTracer,
    Task,
    TaskSpan,
    utc_now,
)
from intelligence_layer.core.tracer.persistent_tracer import TracerLogEntryFailed


@fixture
def complete(
    luminous_control_model: LuminousControlModel,
) -> Task[CompleteInput, CompleteOutput]:
    return luminous_control_model.complete_task()


@fixture
def open_telemetry_tracer() -> tuple[str, OpenTelemetryTracer]:
    service_name = "test-service"
    url = "http://localhost:16686/api/traces?service=" + service_name
    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)
    processor = BatchSpanProcessor(OTLPSpanExporter())
    provider.add_span_processor(processor)
    openTracer = OpenTelemetryTracer(trace.get_tracer("intelligence-layer"))
    return (url, openTracer)


def test_composite_tracer_id_consistent_across_children(
    file_tracer: FileTracer,
) -> None:
    input = "input"
    tracer1 = InMemoryTracer()

    TestTask().run(input, CompositeTracer([tracer1]))
    assert isinstance(tracer1.entries[0], InMemorySpan)
    assert tracer1.entries[0].id() == tracer1.entries[0].entries[0].id()


def test_tracer_id_exists_for_all_children_of_task_span() -> None:
    tracer = InMemoryTracer()
    parent_span = tracer.task_span("child", "input", trace_id="ID")
    parent_span.span("child2")

    assert isinstance(tracer.entries[0], InMemorySpan)
    assert tracer.entries[0].id() == "ID"

    assert tracer.entries[0].entries[0].id() == tracer.entries[0].id()

    parent_span.task_span("child3", "input")
    assert tracer.entries[0].entries[1].id() == tracer.entries[0].id()


def test_tracer_id_exists_for_all_children_of_span() -> None:
    tracer = InMemoryTracer()
    parent_span = tracer.span("child", trace_id="ID")
    parent_span.span("child2")

    assert isinstance(tracer.entries[0], InMemorySpan)
    assert tracer.entries[0].id() == "ID"
    assert tracer.entries[0].entries[0].id() == tracer.entries[0].id()

    parent_span.task_span("child3", "input")
    assert tracer.entries[0].entries[1].id() == tracer.entries[0].id()


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


def test_task_automatically_logs_input_and_output(
    complete: Task[CompleteInput, CompleteOutput],
) -> None:
    tracer = InMemoryTracer()
    input = CompleteInput(prompt=Prompt.from_text("test"))
    output = complete.run(input=input, tracer=tracer)

    assert len(tracer.entries) == 1
    task_span = tracer.entries[0]
    assert isinstance(task_span, InMemoryTaskSpan)
    assert task_span.name == type(complete).__name__
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


def test_composite_tracer(complete: Task[CompleteInput, CompleteOutput]) -> None:
    tracer1 = InMemoryTracer()
    tracer2 = InMemoryTracer()
    input = CompleteInput(prompt=Prompt.from_text("test"))
    complete.run(input=input, tracer=CompositeTracer([tracer1, tracer2]))

    assert tracer1 == tracer2


class TestSubTask(Task[None, None]):
    def do_run(self, input: None, task_span: TaskSpan) -> None:
        task_span.log("subtask", "value")


class TestTask(Task[str, str]):
    sub_task = TestSubTask()

    def do_run(self, input: str, task_span: TaskSpan) -> str:
        with task_span.span("span") as sub_span:
            sub_span.log("message", "a value")
            self.sub_task.run(None, sub_span)
        self.sub_task.run(None, task_span)

        return "output"


@fixture
def file_tracer(tmp_path: Path) -> FileTracer:
    return FileTracer(tmp_path / "log.log")


def test_file_tracer(file_tracer: FileTracer) -> None:
    input = "input"
    expected = InMemoryTracer()

    TestTask().run(input, CompositeTracer([expected, file_tracer]))

    log_tree = file_tracer.trace()
    assert log_tree == expected


def test_file_tracer_retrieves_correct_trace(file_tracer: FileTracer) -> None:
    input = "input"
    expected = InMemoryTracer()
    compositeTracer = CompositeTracer([expected, file_tracer])
    TestTask().run(input, compositeTracer, "ID1")
    TestTask().run(input, file_tracer, "ID2")
    log_tree = file_tracer.trace("ID1")
    assert log_tree == expected


def test_file_tracer_handles_tracer_log_entry_failed_exception(
    file_tracer: FileTracer,
) -> None:
    file_tracer._log_entry = Mock(  # type: ignore[method-assign]
        side_effect=[TracerLogEntryFailed("Hi I am an error", "21"), None]
    )

    try:
        file_tracer.task_span(
            task_name="mock_task_name", input="42", timestamp=None, trace_id="21"
        )
    except Exception as exception:
        assert False, f"'Unexpected exception: {exception}"


def test_file_tracer_raises_non_log_entry_failed_exceptions(
    file_tracer: FileTracer,
) -> None:
    file_tracer._log_entry = Mock(side_effect=[Exception("Hi I am an error", "21")])  # type: ignore[method-assign]
    with pytest.raises(Exception):
        file_tracer.task_span(
            task_name="mock_task_name", input="42", timestamp=None, trace_id="21"
        )


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


def test_in_memory_tracer_trace_viewer_doesnt_crash_if_it_cant_reach() -> None:
    # note that this test sets the environment variable, which might
    # become a problem with multi-worker tests
    ENV_VARIABLE_NAME = "TRACE_VIEWER_URL"
    # ensure that the code works even with the variable is not set
    with set_env(ENV_VARIABLE_NAME, None):
        expected = InMemoryTracer()
        expected._ipython_display_()


def test_open_telemetry_tracer_check_consistency_in_trace_ids(
    open_telemetry_tracer: tuple[str, OpenTelemetryTracer]
) -> None:
    tracing_service, tracer = open_telemetry_tracer
    expected_trace_id = tracer.ensure_id(None)
    TestTask().run("test-input", tracer, trace_id=expected_trace_id)
    trace = _get_trace_by_id(tracing_service, expected_trace_id)

    assert trace is not None
    assert _get_trace_id_from_trace(trace) == expected_trace_id
    spans = trace["spans"]
    assert len(spans) == 4
    for span in spans:
        assert _get_trace_id_from_span(span) == expected_trace_id


def test_open_telemetry_tracer_loggs_input_and_output(
    open_telemetry_tracer: tuple[str, OpenTelemetryTracer],
    complete: Task[CompleteInput, CompleteOutput],
) -> None:
    tracing_service, tracer = open_telemetry_tracer
    input = CompleteInput(prompt=Prompt.from_text("test"))
    trace_id = tracer.ensure_id(None)
    complete.run(input, tracer, trace_id)
    trace = _get_trace_by_id(tracing_service, trace_id)

    assert trace is not None

    spans = trace["spans"]
    assert spans is not []

    task_span = next((span for span in spans if span["references"] == []), None)
    assert task_span is not None

    tags = task_span["tags"]
    open_tel_input_tag = [tag for tag in tags if tag["key"] == "input"]
    assert len(open_tel_input_tag) == 1

    open_tel_output_tag = [tag for tag in tags if tag["key"] == "output"]
    assert len(open_tel_output_tag) == 1


def _get_trace_by_id(tracing_service: str, wanted_trace_id: str) -> Optional[Any]:
    request_timeout_in_seconds = 10
    traces = _get_current_traces(tracing_service)
    if traces:
        for current_trace in traces:
            trace_id = _get_trace_id_from_trace(current_trace)
            if trace_id == wanted_trace_id:
                return trace

    request_start = time.time()
    while time.time() - request_start < request_timeout_in_seconds:
        traces = _get_current_traces(tracing_service)
        if traces:
            for current_trace in traces:
                trace_id = _get_trace_id_from_trace(current_trace)
                if trace_id == wanted_trace_id:
                    return current_trace
        time.sleep(0.1)
    return None


def _get_current_traces(tracing_service: str) -> Any:
    response = requests.get(tracing_service)
    response_text = json.loads(response.text)
    return response_text["data"]


def _get_trace_id_from_trace(trace: Any) -> Optional[str]:
    spans = trace["spans"]
    if not spans:
        return None
    return _get_trace_id_from_span(spans[0])


def _get_trace_id_from_span(span: Any) -> Optional[str]:
    tags = span["tags"]
    if not tags:
        return None
    trace_id_tag = next(tag for tag in tags if tag["key"] == "trace_id")
    return str(trace_id_tag["value"])
