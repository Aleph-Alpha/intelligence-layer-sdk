from datetime import datetime
from aleph_alpha_client import Prompt
from aleph_alpha_client.aleph_alpha_client import Client
from aleph_alpha_client.completion import CompletionRequest

from intelligence_layer.task import DebugLogger, JsonDebugLogger, LogEntry
from intelligence_layer.completion import Completion, CompletionInput


def test_debug_add_log_entries() -> None:
    logger = JsonDebugLogger(name="test")
    logger.log("Test", "message")

    assert len(logger.logs) == 1
    log = logger.logs[0]
    assert isinstance(log, LogEntry)
    assert log.message == "Test"
    assert log.value == "message"
    assert log.timestamp < datetime.utcnow()


def test_can_add_child_debug_logger() -> None:
    logger = JsonDebugLogger(name="parent")

    child_logger = logger.child_logger("child")

    assert len(logger.logs) == 1

    log = logger.logs[0]
    assert isinstance(log, DebugLogger)
    assert log.name == "child"
    assert len(log.logs) == 0


def test_can_add_parent_and_child_logs() -> None:
    parent = JsonDebugLogger(name="parent")
    parent.log("One", 1)
    child = parent.child_logger("child")
    child.log("Two", 2)

    assert isinstance(parent.logs[0], LogEntry)
    assert isinstance(parent.logs[1], DebugLogger)
    assert isinstance(parent.logs[1].logs[0], LogEntry)


def test_task_automatically_logs_input_and_output(client: Client) -> None:
    logger = JsonDebugLogger(name="completion")
    input = CompletionInput(
        request=CompletionRequest(prompt=Prompt.from_text("test")),
        model="luminous-base",
    )
    Completion(client=client).run(input=input, logger=logger)

    assert len(logger.logs) == 2
    assert isinstance(logger.logs[0], LogEntry) and logger.logs[0].message == "Input"
    assert isinstance(logger.logs[1], LogEntry) and logger.logs[1].message == "Output"
