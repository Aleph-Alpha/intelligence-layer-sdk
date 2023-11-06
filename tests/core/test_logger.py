from datetime import datetime
from json import loads
from pathlib import Path
from pprint import pprint
from typing import Any, Mapping, Optional
from uuid import UUID

from pydantic import BaseModel, Field, SerializeAsAny
from pytest import fixture
from intelligence_layer.core.logger import (
    DebugLogger,
    EndSpan,
    EndTask,
    InMemoryDebugLogger,
    InMemorySpan,
    InMemoryTaskSpan,
    FileDebugLogger,
    LogEntry,
    LogLine,
    PlainEntry,
    StartSpan,
    StartTask,
)
from intelligence_layer.core.task import Task


class TestSubTask(Task[None, None]):
    def run(self, input: None, logger: DebugLogger) -> None:
        logger.log("subtask", "value")


class TestTask(Task[str, str]):
    sub_task = TestSubTask()

    def run(self, input: str, logger: DebugLogger) -> str:
        with logger.span("span") as span_logger:
            span_logger.log("message", "a value")
            self.sub_task.run(None, span_logger)
        self.sub_task.run(None, logger)

        return "output"


@fixture
def file_debug_log(tmp_path: Path) -> FileDebugLogger:
    return FileDebugLogger(tmp_path / "log.log")


def test_file_debug_logger(file_debug_log: FileDebugLogger) -> None:
    input = "input"

    output = TestTask().run(input, file_debug_log)

    log_tree = parse_log(file_debug_log._log_file_path)
    expected = InMemoryDebugLogger(name="")
    task = InMemoryTaskSpan(
        name="TestTask",
        end_timestamp=None,
        input=input,
        output=output,
    )
    span = InMemorySpan(name="span", end_timestamp=None)
    span.logs.append(
        LogEntry(message="message", value="a value", timestamp=FIX_TIMESTAMP)
    )
    sub_task = InMemoryTaskSpan(
        name="TestSubTask",
        end_timestamp=None,
        input=None,
        output=None,
    )
    sub_task.logs.append(
        LogEntry(message="subtask", value="value", timestamp=FIX_TIMESTAMP)
    )
    span.logs.append(sub_task)
    task.logs.append(span)
    sub_task = InMemoryTaskSpan(
        name="TestSubTask",
        end_timestamp=None,
        input=None,
        output=None,
    )
    sub_task.logs.append(
        LogEntry(message="subtask", value="value", timestamp=FIX_TIMESTAMP)
    )
    task.logs.append(sub_task)

    expected.logs.append(task)
    assert log_tree == expected


def parse_log(log_path: Path) -> InMemoryDebugLogger:
    tree_builder = TreeBuilder()
    with log_path.open("r") as f:
        for line in f:
            json_line = loads(line)
            log_line = LogLine.model_validate(json_line)
            if log_line.entry_type == StartTask.__name__:
                tree_builder.start_task(log_line)
            elif log_line.entry_type == EndTask.__name__:
                tree_builder.end_task(log_line)
            elif log_line.entry_type == StartSpan.__name__:
                tree_builder.start_span(log_line)
            elif log_line.entry_type == EndSpan.__name__:
                tree_builder.end_span(log_line)
            elif log_line.entry_type == PlainEntry.__name__:
                tree_builder.plain_entry(log_line)
            else:
                raise RuntimeError(f"Unexpected entry_type in {log_line}")
    assert tree_builder.root
    return tree_builder.root


class TreeBuilder(BaseModel):
    root: InMemoryDebugLogger = InMemoryDebugLogger(name="")
    loggers: dict[UUID, InMemoryDebugLogger] = Field(default_factory=dict)
    tasks: dict[UUID, InMemoryTaskSpan] = Field(default_factory=dict)
    spans: dict[UUID, InMemorySpan] = Field(default_factory=dict)

    def start_task(self, log_line: LogLine) -> None:
        start_task = StartTask.model_validate(log_line.entry)
        child = InMemoryTaskSpan(
            name=start_task.name,
            input=start_task.input,
        )
        self.loggers[start_task.uuid] = child
        self.tasks[start_task.uuid] = child
        self.loggers.get(start_task.parent, self.root).logs.append(child)

    def end_task(self, log_line: LogLine) -> None:
        end_task = EndTask.model_validate(log_line.entry)
        task_span = self.tasks[end_task.uuid]
        task_span.end_timestamp = None  # end_task.end
        task_span.record_output(end_task.output)

    def start_span(self, log_line: LogLine) -> None:
        start_span = StartSpan.model_validate(log_line.entry)
        child = InMemorySpan(
            name=start_span.name,
        )
        self.loggers[start_span.uuid] = child
        self.spans[start_span.uuid] = child
        self.loggers.get(start_span.parent, self.root).logs.append(child)

    def end_span(self, log_line: LogLine) -> None:
        end_span = EndSpan.model_validate(log_line.entry)
        span = self.spans[end_span.uuid]
        span.end_timestamp = None  # end_span.end

    def plain_entry(self, log_line: LogLine) -> None:
        plain_entry = PlainEntry.model_validate(log_line.entry)
        entry = LogEntry(
            message=plain_entry.message,
            value=plain_entry.value,
            timestamp=FIX_TIMESTAMP,
        )
        self.loggers[plain_entry.parent].logs.append(entry)


FIX_TIMESTAMP = datetime.utcnow()
