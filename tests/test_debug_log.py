from datetime import datetime
from aleph_alpha_client import Prompt
from aleph_alpha_client.aleph_alpha_client import Client
from aleph_alpha_client.completion import CompletionRequest

from intelligence_layer.task import (
    DebugLogger,
    InMemoryDebugLogger,
    LogEntry,
    TaskLogger,
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
    assert isinstance(log, DebugLogger)
    assert log.name == "child"
    assert len(log.logs) == 0


def test_can_add_parent_and_child_logs() -> None:
    parent = InMemoryDebugLogger(name="parent")
    parent.log("One", 1)
    child = parent.child_logger("child")
    child.log("Two", 2)

    assert isinstance(parent.logs[0], LogEntry)
    assert isinstance(parent.logs[1], DebugLogger)
    assert isinstance(parent.logs[1].logs[0], LogEntry)


def test_task_automatically_logs_input_and_output(client: Client) -> None:
    logger = InMemoryDebugLogger(name="completion")
    input = RawCompletionInput(
        request=CompletionRequest(prompt=Prompt.from_text("test")),
        model="luminous-base",
    )
    output = RawCompletion(client=client).run(input=input, logger=logger)

    assert len(logger.logs) == 1
    task_logger = logger.logs[0]
    assert isinstance(task_logger, TaskLogger)
    assert task_logger.name == "RawCompletion"
    assert task_logger.parent_uuid == logger.uuid
    assert task_logger.input == input
    assert task_logger.output == output
    assert task_logger.start_timestamp and task_logger.end_timestamp
    assert task_logger.start_timestamp < task_logger.end_timestamp
