from datetime import datetime

from aleph_alpha_client import Prompt
from aleph_alpha_client.aleph_alpha_client import Client
from aleph_alpha_client.completion import CompletionRequest

from intelligence_layer.core.complete import Complete, CompleteInput
from intelligence_layer.core.tracer import (
    CompositeTracer,
    InMemorySpan,
    InMemoryTaskSpan,
    InMemoryTracer,
    LogEntry,
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


def test_task_automatically_logs_input_and_output(client: Client) -> None:
    tracer = InMemoryTracer()
    input = CompleteInput(
        request=CompletionRequest(prompt=Prompt.from_text("test")),
        model="luminous-base",
    )
    output = Complete(client=client).run(input=input, tracer=tracer)

    assert len(tracer.entries) == 1
    task_span = tracer.entries[0]
    assert isinstance(task_span, InMemoryTaskSpan)
    assert task_span.name == "Complete"
    assert task_span.input == input
    assert task_span.output == output
    assert task_span.start_timestamp and task_span.end_timestamp
    assert task_span.start_timestamp < task_span.end_timestamp


def test_tracer_can_set_custom_start_time_for_log_entry() -> None:
    tracer = InMemoryTracer()
    timestamp = datetime.utcnow()

    with tracer.span("span") as span:
        span.log("log", "message", timestamp)

    assert isinstance(tracer.entries[0], InMemorySpan)
    assert isinstance(tracer.entries[0].entries[0], LogEntry)
    assert tracer.entries[0].entries[0].timestamp == timestamp


def test_tracer_can_set_custom_start_time_for_span() -> None:
    tracer = InMemoryTracer()
    start = datetime.utcnow()

    span = tracer.span("span", start)

    assert span.start_timestamp == start


def test_span_sets_end_timestamp() -> None:
    tracer = InMemoryTracer()
    start = datetime.utcnow()

    span = tracer.span("span", start)
    span.end()

    assert span.end_timestamp and span.start_timestamp <= span.end_timestamp


def test_span_only_updates_end_timestamp_once() -> None:
    tracer = InMemoryTracer()

    span = tracer.span("span")
    end = datetime.utcnow()
    span.end(end)
    span.end()

    assert span.end_timestamp == end


def test_composite_tracer(client: Client) -> None:
    tracer1 = InMemoryTracer()
    tracer2 = InMemoryTracer()
    input = CompleteInput(
        request=CompletionRequest(prompt=Prompt.from_text("test")),
        model="luminous-base",
    )
    Complete(client=client).run(input=input, tracer=CompositeTracer([tracer1, tracer2]))

    assert tracer1 == tracer2
