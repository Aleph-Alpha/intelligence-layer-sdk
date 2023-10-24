from datetime import datetime
from aleph_alpha_client import Prompt
from aleph_alpha_client.aleph_alpha_client import Client
from aleph_alpha_client.completion import CompletionRequest

from intelligence_layer.task import (
    InMemoryDebugLogger,
    InMemoryTaskLogger,
    LogEntry,
)
from intelligence_layer.completion import RawCompletion, RawCompletionInput


def test_debug_add_log_entries() -> None:
    before_log_time = datetime.utcnow()
    logger = InMemoryDebugLogger(name="test")
    logger.log("Test", "message")

    assert len(logger.logs) == 1
    log = logger.logs[0]
    assert isinstance(log, LogEntry)
    assert log.message == "Test"
    assert log.value == "message"
    assert before_log_time <= log.timestamp <= datetime.utcnow()


def test_can_add_child_debug_logger() -> None:
    logger = InMemoryDebugLogger(name="parent")

    child_logger = logger.child_logger("child")

    assert len(logger.logs) == 1

    log = logger.logs[0]
    assert isinstance(log, InMemoryDebugLogger)
    assert log.name == "child"
    assert len(log.logs) == 0


def test_can_add_parent_and_child_logs() -> None:
    parent = InMemoryDebugLogger(name="parent")
    parent.log("One", 1)
    child = parent.child_logger("child")
    child.log("Two", 2)

    assert isinstance(parent.logs[0], LogEntry)
    assert isinstance(parent.logs[1], InMemoryDebugLogger)
    assert isinstance(parent.logs[1].logs[0], LogEntry)


def test_task_automatically_logs_input_and_output(client: Client) -> None:
    logger = InMemoryDebugLogger(name="completion")
    input = RawCompletionInput(
        request=CompletionRequest(prompt=Prompt.from_text("test")),
        model="luminous-base",
    )
    output = RawCompletion(client=client).run(input=input, logger=logger)

    assert len(logger.logs) == 1
    task_span = logger.logs[0]
    assert isinstance(task_span, InMemoryTaskLogger)
    assert task_span.name == "RawCompletion"
    assert task_span.input == input
    assert task_span.output == output
    assert task_span.start_timestamp and task_span.end_timestamp
    assert task_span.start_timestamp < task_span.end_timestamp
