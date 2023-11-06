from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from datetime import datetime
from json import dumps
from pathlib import Path
from types import TracebackType
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, RootModel, SerializeAsAny
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree

from typing_extensions import TypeAliasType, Self

from typing import (
    TYPE_CHECKING,
    Any,
    Mapping,
    Optional,
    Sequence,
    Union,
)

if TYPE_CHECKING:
    PydanticSerializable = (
        int
        | float
        | str
        | Sequence["PydanticSerializable"]
        | Mapping[str, "PydanticSerializable"]
        | None
        | bool
        | BaseModel
        | UUID
    )
else:
    PydanticSerializable = TypeAliasType(
        "PydanticSerializable",
        int
        | float
        | str
        | Sequence["PydanticSerializable"]
        | Mapping[str, "PydanticSerializable"]
        | None
        | bool
        | BaseModel
        | UUID,
    )


class DebugLogger(ABC):
    """A protocol for instrumenting `Task`s with structured logging.

    A logger needs to provide a way to collect an individual log, which should be serializable, and
    a way to generate nested loggers, so that sub-tasks can emit logs that are grouped together.

    Each `DebugLogger` is given a `name` to distinguish them from each other, and for nested logs.

    Implementations of how logs are collected and stored may differ. Refer to the individual
    documentation of each implementation to see how to use the resulting logger.
    """

    @abstractmethod
    def log(
        self,
        message: str,
        value: PydanticSerializable,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a log of relevant information as part of a step within a task.

        By default, the `Input` and `Output` of each `Task` are logged automatically, but you can
        log anything else that seems relevant to understanding the output of a given task.

        Args:
            message: A description of the value you are logging, such as the step in the task this
                is related to.
            value: The relevant data you want to log. Can be anything that is serializable by
                Pydantic, which gives the loggers flexibility in how they store and emit the logs.
            timestamp: optional override of the timestamp. Otherwise should default to now
        """
        ...

    @abstractmethod
    def span(self, name: str, timestamp: Optional[datetime] = None) -> "Span":
        """Generate a span from the current logging instance.

        Each logger implementation can decide on how it wants to represent this, but they should
        all allow for representing logs of a child task within the scope of the current task.

        Args:
            name: A descriptive name of what this span will contain logs about.
            timestamp: optional override of the starting timestamp. Otherwise should default to now

        Returns:
            An instance of something that meets the protocol of Span.
        """
        ...

    @abstractmethod
    def task_span(
        self,
        task_name: str,
        input: PydanticSerializable,
        timestamp: Optional[datetime] = None,
    ) -> "TaskSpan":
        """Generate a task-specific span from the current logging instance.

        Each logger implementation can decide on how it wants to represent this, but they should
        all allow for representing logs of a span within the context of a parent span.

        Args:
            task_name: The name of the task that is being logged
            input: The input for the task that is being logged.
            timestamp: optional override of the starting timestamp. Otherwise should default to now

        Returns:
            An instance of something that also meets the protocol of DebugLogger. Most likely, it
            will create an instance of the same type, but this is dependent on the actual
            implementation.
        """
        ...


class Span(DebugLogger, AbstractContextManager["Span"]):
    """A protocol for instrumenting logs nested within a span of time. Groups logs by some logical
    step.

    The implementation should also be a Context Manager, to capture the span of duration of
    execution.

    Implementations of how logs are collected and stored may differ. Refer to the individual
    documentation of each implementation to see how to use the resulting logger.
    """

    def __enter__(self) -> Self:
        return self

    @abstractmethod
    def end(self, timestamp: Optional[datetime] = None) -> None:
        """Marks the span as done.

        Args:
            timestamp: Optional override of the timestamp, otherwise should be set to now.
        """
        ...

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.end()


class TaskSpan(Span, AbstractContextManager["TaskSpan"]):
    """A protocol for instrumenting a `Task`'s input, output, and nested logs.

    Most likely, generating this task logger will capture the `Task`'s input, as well as the task
    name.

    The implementation should also be a Context Manager, to capture the span of duration of
    task execution.

    Implementations of how logs are collected and stored may differ. Refer to the individual
    documentation of each implementation to see how to use the resulting logger.
    """

    def __enter__(self) -> Self:
        return self

    @abstractmethod
    def record_output(self, output: PydanticSerializable) -> None:
        """Record a `Task`'s output. Since a Context Manager can't provide this in the `__exit__`
        method, output should be captured once it is generated.

        This should be handled automatically within the execution of the task.

        Args:
            output: The output of the task that is being logged.
        """
        ...


class CompositeLogger(TaskSpan):
    """A `DebugLogger` that allows for recording to multiple loggers simultaneously.

    Each log-entry and span will be forwarded to all subloggers.

    Args:
        loggers: Loggers that will be forwarded all log and span calls.
    """

    def __init__(self, loggers: Sequence[DebugLogger]) -> None:
        self.loggers = loggers

    def log(
        self,
        message: str,
        value: PydanticSerializable,
        timestamp: Optional[datetime] = None,
    ) -> None:
        timestamp = timestamp or datetime.utcnow()
        for logger in self.loggers:
            logger.log(message, value, timestamp)

    def span(
        self, name: str, timestamp: Optional[datetime] = None
    ) -> "CompositeLogger":
        """Generate a sub-logger from the current logging instance.

        Args:
            name: A descriptive name of what this child logger will contain logs about.
            timestamp: Optional timestamp override for the start timestamp of the span.

        Returns:
            Another `CompositeLogger`
        """
        timestamp = timestamp or datetime.utcnow()
        return CompositeLogger(
            [logger.span(name, timestamp) for logger in self.loggers]
        )

    def task_span(
        self,
        task_name: str,
        input: PydanticSerializable,
        timestamp: Optional[datetime] = None,
    ) -> "CompositeLogger":
        """Generate a task-specific span from the current logging instance.

        Args:
            task_name: The name of the task that is being logged
            input: The input for the task that is being logged.
            timestamp: Optional timestamp override for the start timestamp of the span.

        Returns:
            A `CompositeLogger`
        """

        timestamp = timestamp or datetime.utcnow()
        return CompositeLogger(
            [logger.task_span(task_name, input, timestamp) for logger in self.loggers]
        )

    def end(self, timestamp: Optional[datetime] = None) -> None:
        """Marks the span as done.

        Args:
            timestamp: Optional override of the timestamp, otherwise should be set to now.
        """
        timestamp = timestamp or datetime.utcnow()
        for logger in self.loggers:
            if isinstance(logger, Span):
                logger.end(timestamp)

    def record_output(self, output: PydanticSerializable) -> None:
        """Record a `Task`'s output. Since a Context Manager can't provide this in the `__exit__`
        method, output should be captured once it is generated.

        This should be handled automatically within the execution of the task.

        Args:
            output: The output of the task that is being logged.
        """
        for logger in self.loggers:
            if isinstance(logger, TaskSpan):
                logger.record_output(output)


class NoOpDebugLogger(DebugLogger):
    """A no-op logger. Useful for cases, like testing, where a logger is needed for a task, but you
    don't have a need to collect or inspect the actual logs.

    All calls to `log` won't actually do anything.
    """

    def log(
        self,
        message: str,
        value: PydanticSerializable,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """The `NoOpDebugLogger` does not do anything.

        Args:
            message: A description of the value you are logging, such as the step in the task this
                is related to.
            value: The relevant data you want to log. Can be anything that is serializable by
                Pydantic, which gives the loggers flexibility in how they store and emit the logs.
            timestamp: Optional timestamp override for the log entry.
        """
        pass

    def span(self, name: str, timestamp: Optional[datetime] = None) -> "NoOpTaskSpan":
        """Generate a sub-logger from the current logging instance.

        Args:
            name: A descriptive name of what this child logger will contain logs about.
            timestamp: Optional timestamp override for the log entry.

        Returns:
            Another `NoOpDebugLogger`
        """
        return NoOpTaskSpan()

    def task_span(
        self,
        task_name: str,
        input: PydanticSerializable,
        timestamp: Optional[datetime] = None,
    ) -> "NoOpTaskSpan":
        """Generate a task-specific span from the current logging instance.


        Args:
            task_name: The name of the task that is being logged
            input: The input for the task that is being logged.

        Returns:
            A `NoOpTaskSpan`
        """

        return NoOpTaskSpan()


class NoOpTaskSpan(NoOpDebugLogger, TaskSpan):
    def record_output(self, output: PydanticSerializable) -> None:
        """The `NoOpTaskSpan` does not do anything.

        Args:
            output: The output of the task that is being logged.
        """
        pass

    def end(self, timestamp: Optional[datetime] = None) -> None:
        """Marks the span as done.

        Args:
            timestamp: Optional override of the timestamp, otherwise should be set to now.
        """
        pass


class JsonSerializer(RootModel[PydanticSerializable]):
    root: SerializeAsAny[PydanticSerializable]


def _render_log_value(value: PydanticSerializable, title: str) -> Panel:
    value = value if isinstance(value, BaseModel) else JsonSerializer(root=value)
    return Panel(
        Syntax(
            value.model_dump_json(indent=2, exclude_defaults=True),
            "json",
            word_wrap=True,
        ),
        title=title,
    )


class LogEntry(BaseModel):
    """An individual log entry, currently used to represent individual logs by the
    `InMemoryDebugLogger`.

    Attributes:
        message: A description of the value you are logging, such as the step in the task this
            is related to.
        value: The relevant data you want to log. Can be anything that is serializable by
            Pydantic, which gives the loggers flexibility in how they store and emit the logs.
        timestamp: The time that the log was emitted.
    """

    message: str
    value: SerializeAsAny[PydanticSerializable]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def _rich_render_(self) -> Panel:
        """Renders the debug log via classes in the `rich` package"""
        return _render_log_value(self.value, self.message)

    def _ipython_display_(self) -> None:
        """Default rendering for Jupyter notebooks"""
        from rich import print

        print(self._rich_render_())


class InMemoryDebugLogger(BaseModel, DebugLogger):
    """Collects log entries in a nested structure, and keeps them in memory.

    If desired, the structure is serializable with Pydantic, so you can write out the JSON
    representation to a file, or return via an API, or something similar.

    Attributes:
        name: A descriptive name of what the logger contains log entries about.
        logs: A sequential list of log entries and/or nested InMemoryDebugLoggers with their own
            log entries.
    """

    name: str
    logs: list[Union[LogEntry, "InMemorySpan", "InMemoryTaskSpan"]] = []

    def log(
        self,
        message: str,
        value: PydanticSerializable,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a log of relevant information as part of a step within a task.

        By default, the `Input` and `Output` of each `Task` are logged automatically, but you can
        log anything else that seems relevant to understanding the output of a given task.

        Args:
            message: A description of the value you are logging, such as the step in the task this
                is related to.
            value: The relevant data you want to log. Can be anything that is serializable by
                Pydantic, which gives the loggers flexibility in how they store and emit the logs.
            timestamp: Optional timestamp override for the log entry. Otherwise will be set to now
        """
        self.logs.append(
            LogEntry(
                message=message, value=value, timestamp=timestamp or datetime.utcnow()
            )
        )

    def span(self, name: str, timestamp: Optional[datetime] = None) -> "InMemorySpan":
        """Generate a sub-logger from the current logging instance.

        Args:
            name: A descriptive name of what this child logger will contain logs about.
            timestamp: Optional starting timestamp of the span. Otherwise defaults to now

        Returns:
            A nested `InMemoryDebugLogger` that is stored in a nested position as part of the parent
            logger.
        """
        child = InMemorySpan(name=name, start_timestamp=timestamp or datetime.utcnow())
        self.logs.append(child)
        return child

    def task_span(
        self,
        task_name: str,
        input: PydanticSerializable,
        timestamp: Optional[datetime] = None,
    ) -> "InMemoryTaskSpan":
        """Generate a task-specific span from the current logging instance.


        Args:
            task_name: The name of the task that is being logged
            input: The input for the task that is being logged.
            timestamp: optional override of the starting timestamp. Otherwise should default to now

        Returns:
            A nested `InMemoryTaskSpan` that is stored in a nested position as part of the parent
            logger
        """

        child = InMemoryTaskSpan(
            name=task_name, input=input, start_timestamp=timestamp or datetime.utcnow()
        )
        self.logs.append(child)
        return child

    def _rich_render_(self) -> Tree:
        """Renders the debug log via classes in the `rich` package"""
        tree = Tree(label=self.name)

        for log in self.logs:
            tree.add(log._rich_render_())

        return tree

    def _ipython_display_(self) -> None:
        """Default rendering for Jupyter notebooks"""
        from rich import print

        print(self._rich_render_())


class InMemorySpan(InMemoryDebugLogger, Span):
    start_timestamp: datetime = Field(default_factory=datetime.utcnow)
    end_timestamp: Optional[datetime] = None

    def end(self, timestamp: Optional[datetime] = None) -> None:
        """Marks the span as done.

        Args:
            timestamp: Optional override of the timestamp, otherwise should be set to now.
        """
        if not self.end_timestamp:
            self.end_timestamp = timestamp or datetime.utcnow()

    def _rich_render_(self) -> Tree:
        """Renders the debug log via classes in the `rich` package"""
        tree = Tree(label=self.name)

        for log in self.logs:
            tree.add(log._rich_render_())

        return tree


class InMemoryTaskSpan(InMemorySpan, TaskSpan):
    input: SerializeAsAny[PydanticSerializable]
    output: Optional[SerializeAsAny[PydanticSerializable]] = None

    def record_output(self, output: PydanticSerializable) -> None:
        """Record a `Task`'s output. Since a Context Manager can't provide this in the `__exit__`
        method, output should be captured once it is generated.

        This should be handled automatically within the execution of the task.

        Args:
            output: The output of the task that is being logged.
        """
        self.output = output

    def _rich_render_(self) -> Tree:
        """Renders the debug log via classes in the `rich` package"""
        tree = Tree(label=self.name)

        tree.add(_render_log_value(self.input, "Input"))

        for log in self.logs:
            tree.add(log._rich_render_())

        tree.add(_render_log_value(self.output, "Output"))

        return tree


# Required for sphinx, see also: https://docs.pydantic.dev/2.4/errors/usage_errors/#class-not-fully-defined
InMemorySpan.model_rebuild()
InMemoryDebugLogger.model_rebuild()


class StartTask(BaseModel):
    """Represents the payload/entry of a log-line indicating that a `TaskSpan` was opened through `DebugLogger.task_span`.

    Attributes:
        uuid: A unique id for the opened `TaskSpan`.
        parent: The unique id of the parent element of opened `TaskSpan`.
            This could refer to either a surrounding `TaskSpan`, `Span` or the top-level `DebugLogger`.
        name: The name of the task.
        start: The timestamp when this `Task` was started (i.e. `run` was called).
        input: The `Input` (i.e. parameter for `run`) the `Task` was started with.
    """

    uuid: UUID
    parent: UUID
    name: str
    start: datetime
    input: SerializeAsAny[Any]


class EndTask(BaseModel):
    """Represents the payload/entry of a log-line that indicates that a `TaskSpan` ended (i.e. the context-manager exited).

    Attributes:
        uuid: The uuid of the corresponding `StartTask`.
        end: the timestamp when this `Task` completed (i.e. `run` returned).
        output: the `Output` (i.e. return value of `run`) the `Task` returned.
    """

    uuid: UUID
    end: datetime
    output: SerializeAsAny[Any]


class StartSpan(BaseModel):
    """Represents the payload/entry of a log-line indicating that a `Span` was opened through `DebugLogger.span`.

    Attributes:
        uuid: A unique id for the opened `Span`.
        parent: The unique id of the parent element of opened `TaskSpan`.
            This could refer to either a surrounding `TaskSpan`, `Span` or the top-level `DebugLogger`.
        name: The name of the task.
        start: The timestamp when this `Span` was started.
    """

    uuid: UUID
    parent: UUID
    name: str
    start: datetime


class EndSpan(BaseModel):
    """Represents the payload/entry of a log-line that indicates that a `Span` ended.

    Attributes:
        uuid: The uuid of the corresponding `StartSpan`.
        end: the timestamp when this `Span` completed.
    """

    uuid: UUID
    end: datetime


class PlainEntry(BaseModel):
    """Represents a plain log-entry created through `DebugLogger.log`.

    Attributes:
        message: the message-parameter of `DebugLogger.log`
        value: the value-parameter of `DebugLogger.log`
        timestamp: the timestamp when `DebugLogger.log` was called.
        parent: The unique id of the parent element of the log.
            This could refer to either a surrounding `TaskSpan`, `Span` or the top-level `DebugLogger`.
    """

    message: str
    value: SerializeAsAny[Any]
    timestamp: datetime
    parent: UUID


class LogLine(BaseModel):
    """Represents a a complete log-line.

    Attributes:
        entry_type: The type of the entry. This is the class-name of one of the classes
            representing a log-entry (e.g. "StartTask").
        entry: The actual entry.

    """

    entry_type: str
    entry: SerializeAsAny[Any]


class FileDebugLogger(DebugLogger):
    """A `DebugLogger` that logs to a file.

    Each log-entry is represented by a JSON object. The information logged allows
    to reconstruct the hierarchical nature of the logs, i.e. all entries have a
    _pointer_ to its parent element in form of a parent attribute containing
    the uuid of the parent.

    Args:
        log_file_path: Denotes the file to log to.

    Attributes:
        uuid: a uuid for the logger. If multiple `FileDebugLogger`s log to the same file
            the child-elements for a logger can be identified by referring to this id as parent.
    """

    def __init__(self, log_file_path: Path) -> None:
        self._log_file_path = log_file_path
        self.uuid = uuid4()

    def log(
        self,
        message: str,
        value: PydanticSerializable,
        timestamp: Optional[datetime] = None,
    ) -> None:
        self._log_entry(
            PlainEntry(
                message=message,
                value=value,
                timestamp=timestamp or datetime.utcnow(),
                parent=self.uuid,
            )
        )

    def _log_entry(self, entry: BaseModel) -> None:
        with self._log_file_path.open("a") as f:
            f.write(
                LogLine(entry_type=type(entry).__name__, entry=entry).model_dump_json()
                + "\n"
            )

    def span(self, name: str, timestamp: Optional[datetime] = None) -> "FileSpan":
        span = FileSpan(self._log_file_path, name)
        self._log_entry(
            StartSpan(
                uuid=span.uuid,
                parent=self.uuid,
                name=name,
                start=timestamp or datetime.utcnow(),
            )
        )
        return span

    def task_span(
        self,
        task_name: str,
        input: PydanticSerializable,
        timestamp: Optional[datetime] = None,
    ) -> "FileTaskSpan":
        task = FileTaskSpan(self._log_file_path, task_name, input)
        self._log_entry(
            StartTask(
                uuid=task.uuid,
                parent=self.uuid,
                name=task_name,
                start=timestamp or datetime.utcnow(),
                input=input,
            )
        )
        return task


class FileSpan(Span, FileDebugLogger):
    """A `Span` created by `FileDebugLogger.span`."""

    end_timestamp: Optional[datetime] = None

    def __init__(self, log_file_path: Path, name: str) -> None:
        super().__init__(log_file_path)

    def end(self, timestamp: Optional[datetime] = None) -> None:
        """Marks the span as done.

        Args:
            timestamp: Optional override of the timestamp, otherwise should be set to now.
        """
        if not self.end_timestamp:
            self.end_timestamp = timestamp or datetime.utcnow()
            self._log_entry(EndSpan(uuid=self.uuid, end=self.end_timestamp))


class FileTaskSpan(TaskSpan, FileSpan):
    """A `TaskSpan` created by `FileDebugLogger.task_span`."""

    output: Optional[PydanticSerializable] = None

    def __init__(
        self, log_file_path: Path, task_name: str, input: PydanticSerializable
    ) -> None:
        super().__init__(log_file_path, task_name)

    def record_output(self, output: PydanticSerializable) -> None:
        self.output = output

    def end(self, timestamp: Optional[datetime] = None) -> None:
        """Marks the span as done.

        Args:
            timestamp: Optional override of the timestamp, otherwise should be set to now.
        """
        if not self.end_timestamp:
            self.end_timestamp = timestamp or datetime.utcnow()
            self._log_entry(
                EndTask(uuid=self.uuid, end=self.end_timestamp, output=self.output)
            )
