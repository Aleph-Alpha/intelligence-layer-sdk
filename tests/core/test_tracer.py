from datetime import datetime

from aleph_alpha_client import Prompt
from aleph_alpha_client.aleph_alpha_client import Client
from aleph_alpha_client.completion import CompletionRequest

from intelligence_layer.core.complete import Complete, CompleteInput
from intelligence_layer.core.tracer import (
    CompositeLogger,
    InMemoryTaskSpan,
    InMemoryTracer,
    LogEntry,
)


def test_debug_add_log_entries() -> None:
    before_log_time = datetime.utcnow()
    logger = InMemoryTracer()
    logger.log("Test", "message")

    assert len(logger.logs) == 1
    log = logger.logs[0]
    assert isinstance(log, LogEntry)
    assert log.message == "Test"
    assert log.value == "message"
    assert before_log_time <= log.timestamp <= datetime.utcnow()


def test_can_add_child_debug_logger() -> None:
    logger = InMemoryTracer()
    logger.span("child")

    assert len(logger.logs) == 1

    log = logger.logs[0]
    assert isinstance(log, InMemoryTracer)
    assert log.name == "child"
    assert len(log.logs) == 0


def test_can_add_parent_and_child_logs() -> None:
    parent = InMemoryTracer()
    parent.log("One", 1)
    with parent.span("child") as child:
        child.log("Two", 2)

    assert isinstance(parent.logs[0], LogEntry)
    assert isinstance(parent.logs[1], InMemoryTracer)
    assert isinstance(parent.logs[1].logs[0], LogEntry)


def test_task_automatically_logs_input_and_output(client: Client) -> None:
    logger = InMemoryTracer()
    input = CompleteInput(
        request=CompletionRequest(prompt=Prompt.from_text("test")),
        model="luminous-base",
    )
    output = Complete(client=client).run(input=input, logger=logger)

    assert len(logger.logs) == 1
    task_span = logger.logs[0]
    assert isinstance(task_span, InMemoryTaskSpan)
    assert task_span.name == "Complete"
    assert task_span.input == input
    assert task_span.output == output
    assert task_span.start_timestamp and task_span.end_timestamp
    assert task_span.start_timestamp < task_span.end_timestamp


def test_logger_can_set_custom_start_time_for_log_entry() -> None:
    logger = InMemoryTracer()
    timestamp = datetime.utcnow()

    logger.log("log", "message", timestamp)

    assert isinstance(logger.logs[0], LogEntry)
    assert logger.logs[0].timestamp == timestamp


def test_logger_can_set_custom_start_time_for_span() -> None:
    logger = InMemoryTracer()
    start = datetime.utcnow()

    span = logger.span("span", start)

    assert span.start_timestamp == start


def test_span_sets_end_timestamp() -> None:
    logger = InMemoryTracer()
    start = datetime.utcnow()

    span = logger.span("span", start)
    span.end()

    assert span.end_timestamp and span.start_timestamp <= span.end_timestamp


def test_span_only_updates_end_timestamp_once() -> None:
    logger = InMemoryTracer()

    span = logger.span("span")
    end = datetime.utcnow()
    span.end(end)
    span.end()

    assert span.end_timestamp == end


def test_composite_logger(client: Client) -> None:
    logger1 = InMemoryTracer()
    logger2 = InMemoryTracer()
    input = CompleteInput(
        request=CompletionRequest(prompt=Prompt.from_text("test")),
        model="luminous-base",
    )
    Complete(client=client).run(input=input, logger=CompositeLogger([logger1, logger2]))

    assert logger1 == logger2
