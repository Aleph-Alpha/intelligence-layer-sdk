from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
)
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, RootModel, SerializeAsAny
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree
from typing_extensions import Self, TypeAliasType

if TYPE_CHECKING:
    PydanticSerializable = (
        int
        | float
        | str
        | Sequence["PydanticSerializable"]
        | set["PydanticSerializable"]
        | frozenset["PydanticSerializable"]
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
        | set["PydanticSerializable"]
        | frozenset["PydanticSerializable"]
        | Mapping[str, "PydanticSerializable"]
        | None
        | bool
        | BaseModel
        | UUID,
    )


class Tracer(ABC):
    """Provides a consistent way to instrument a :class:`Task` with logging for each step of the
    workflow.

    A tracer needs to provide a way to collect an individual log, which should be serializable, and
    a way to generate nested spans, so that sub-tasks can emit logs that are grouped together.

    Implementations of how logs are collected and stored may differ. Refer to the individual
    documentation of each implementation to see how to use the resulting tracer.
    """

    @abstractmethod
    def span(self, name: str, timestamp: Optional[datetime] = None) -> "Span":
        """Generate a span from the current span or logging instance.

        Allows for grouping multiple logs and duration together as a single, logical step in the
        process.

        Each tracer implementation can decide on how it wants to represent this, but they should
        all capture the hierarchical nature of nested spans, as well as the idea of the duration of
        the span.

        Args:
            name: A descriptive name of what this span will contain logs about.
            timestamp: optional override of the starting timestamp. Otherwise should default to now

        Returns:
            An instance of a Span.
        """
        ...

    @abstractmethod
    def task_span(
        self,
        task_name: str,
        input: PydanticSerializable,
        timestamp: Optional[datetime] = None,
    ) -> "TaskSpan":
        """Generate a task-specific span from the current span or logging instance.

        Allows for grouping multiple logs together, as well as the task's specific input, output,
        and duration.

        Each tracer implementation can decide on how it wants to represent this, but they should
        all allow for representing logs of a span within the context of a parent span.

        Args:
            task_name: The name of the task that is being logged
            input: The input for the task that is being logged.
            timestamp: optional override of the starting timestamp. Otherwise should default to now

        Returns:
            An instance of a TaskSpan.
        """
        ...


class Span(Tracer, AbstractContextManager["Span"]):
    """Captures a logical step within the overall workflow

    Logs and other spans can be nested underneath.

    Can also be used as a Context Manager to easily capture the start and end time, and keep the
    span only in scope while it is active.
    """

    def __enter__(self) -> Self:
        return self

    @abstractmethod
    def log(
        self,
        message: str,
        value: PydanticSerializable,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a log of relevant information as part of a step within a task.

        By default, the `Input` and `Output` of each :class:`Task` are logged automatically, but
        you can log anything else that seems relevant to understanding the process of a given task.

        Args:
            message: A description of the value you are logging, such as the step in the task this
                is related to.
            value: The relevant data you want to log. Can be anything that is serializable by
                Pydantic, which gives the tracers flexibility in how they store and emit the logs.
            timestamp: optional override of the timestamp. Otherwise should default to now
        """
        ...

    @abstractmethod
    def end(self, timestamp: Optional[datetime] = None) -> None:
        """Marks the Span as done, with the end time of the span. The Span should be regarded
        as complete, and no further logging should happen with it.

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


class TaskSpan(Span):
    """Specialized span for instrumenting :class:`Task` input, output, and nested spans and logs.

    Generating this TaskSpan should capture the :class:`Task` input, as well as the task name.

    Can also be used as a Context Manager to easily capture the start and end time of the task,
    and keep the span only in scope while it is active
    """

    @abstractmethod
    def record_output(self, output: PydanticSerializable) -> None:
        """Record :class:`Task` output. Since a Context Manager can't provide this in the `__exit__`
        method, output should be captured once it is generated.

        This should be handled automatically within the execution of the task, and it is
        unlikely this would be called directly by you.

        Args:
            output: The output of the task that is being logged.
        """
        ...


TracerVar = TypeVar("TracerVar", bound=Tracer)


class CompositeTracer(Tracer, Generic[TracerVar]):
    """A :class:`Tracer` that allows for recording to multiple tracers simultaneously.

    Each log-entry and span will be forwarded to all subtracers.

    Args:
        tracers: tracers that will be forwarded all subsequent log and span calls.

    Example:
        >>> sub_tracer1 = InMemoryTracer()
        >>> sub_tracer2 = FileTracer("./log.log")
        >>> tracer = CompositeTracer([tracer1, tracer2])
        >>>
        >>> SomeTask.run(input, tracer)
    """

    def __init__(self, tracers: Sequence[TracerVar]) -> None:
        self.tracers = tracers

    def span(
        self, name: str, timestamp: Optional[datetime] = None
    ) -> "CompositeSpan[Span]":
        timestamp = timestamp or datetime.utcnow()
        return CompositeSpan([tracer.span(name, timestamp) for tracer in self.tracers])

    def task_span(
        self,
        task_name: str,
        input: PydanticSerializable,
        timestamp: Optional[datetime] = None,
    ) -> "CompositeTaskSpan":
        timestamp = timestamp or datetime.utcnow()
        return CompositeTaskSpan(
            [tracer.task_span(task_name, input, timestamp) for tracer in self.tracers]
        )


SpanVar = TypeVar("SpanVar", bound=Span)


class CompositeSpan(Generic[SpanVar], CompositeTracer[SpanVar], Span):
    """A :class:`Span` that allows for recording to multiple spans simultaneously.

    Each log-entry and span will be forwarded to all subspans.

    Args:
        tracers: spans that will be forwarded all subsequent log and span calls.
    """

    def log(
        self,
        message: str,
        value: PydanticSerializable,
        timestamp: Optional[datetime] = None,
    ) -> None:
        timestamp = timestamp or datetime.utcnow()
        for tracer in self.tracers:
            tracer.log(message, value, timestamp)

    def end(self, timestamp: Optional[datetime] = None) -> None:
        timestamp = timestamp or datetime.utcnow()
        for tracer in self.tracers:
            tracer.end(timestamp)


class CompositeTaskSpan(CompositeSpan[TaskSpan], TaskSpan):
    """A :class:`TaskSpan` that allows for recording to multiple TaskSpans simultaneously.

    Each log-entry and span will be forwarded to all subspans.

    Args:
        tracers: task spans that will be forwarded all subsequent log and span calls.
    """

    def record_output(self, output: PydanticSerializable) -> None:
        for tracer in self.tracers:
            tracer.record_output(output)


class NoOpTracer(TaskSpan):
    """A no-op tracer.

    Useful for cases, like testing, where a tracer is needed for a task, but you
    don't have a need to collect or inspect the actual logs.

    All calls to `log` won't actually do anything.
    """

    def log(
        self,
        message: str,
        value: PydanticSerializable,
        timestamp: Optional[datetime] = None,
    ) -> None:
        pass

    def span(self, name: str, timestamp: Optional[datetime] = None) -> "NoOpTracer":
        return self

    def task_span(
        self,
        task_name: str,
        input: PydanticSerializable,
        timestamp: Optional[datetime] = None,
    ) -> "NoOpTracer":
        return self

    def record_output(self, output: PydanticSerializable) -> None:
        pass

    def end(self, timestamp: Optional[datetime] = None) -> None:
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
    `InMemoryTracer`.

    Attributes:
        message: A description of the value you are logging, such as the step in the task this
            is related to.
        value: The relevant data you want to log. Can be anything that is serializable by
            Pydantic, which gives the tracers flexibility in how they store and emit the logs.
        timestamp: The time that the log was emitted.
    """

    message: str
    value: SerializeAsAny[PydanticSerializable]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def _rich_render_(self) -> Panel:
        """Renders the trace via classes in the `rich` package"""
        return _render_log_value(self.value, self.message)

    def _ipython_display_(self) -> None:
        """Default rendering for Jupyter notebooks"""
        from rich import print

        print(self._rich_render_())


class InMemoryTracer(BaseModel, Tracer):
    """Collects log entries in a nested structure, and keeps them in memory.

    If desired, the structure is serializable with Pydantic, so you can write out the JSON
    representation to a file, or return via an API, or something similar.

    Attributes:
        name: A descriptive name of what the tracer contains log entries about.
        entries: A sequential list of log entries and/or nested InMemoryTracers with their own
            log entries.
    """

    entries: list[Union[LogEntry, "InMemorySpan", "InMemoryTaskSpan"]] = []

    def span(self, name: str, timestamp: Optional[datetime] = None) -> "InMemorySpan":
        child = InMemorySpan(name=name, start_timestamp=timestamp or datetime.utcnow())
        self.entries.append(child)
        return child

    def task_span(
        self,
        task_name: str,
        input: PydanticSerializable,
        timestamp: Optional[datetime] = None,
    ) -> "InMemoryTaskSpan":
        child = InMemoryTaskSpan(
            name=task_name, input=input, start_timestamp=timestamp or datetime.utcnow()
        )
        self.entries.append(child)
        return child

    def _rich_render_(self) -> Tree:
        """Renders the trace via classes in the `rich` package"""
        tree = Tree(label="Trace")

        for log in self.entries:
            tree.add(log._rich_render_())

        return tree

    def _ipython_display_(self) -> None:
        """Default rendering for Jupyter notebooks"""
        from rich import print

        print(self._rich_render_())


class InMemorySpan(InMemoryTracer, Span):
    name: str
    start_timestamp: datetime = Field(default_factory=datetime.utcnow)
    end_timestamp: Optional[datetime] = None

    def log(
        self,
        message: str,
        value: PydanticSerializable,
        timestamp: Optional[datetime] = None,
    ) -> None:
        self.entries.append(
            LogEntry(
                message=message, value=value, timestamp=timestamp or datetime.utcnow()
            )
        )

    def end(self, timestamp: Optional[datetime] = None) -> None:
        if not self.end_timestamp:
            self.end_timestamp = timestamp or datetime.utcnow()

    def _rich_render_(self) -> Tree:
        """Renders the trace via classes in the `rich` package"""
        tree = Tree(label=self.name)

        for log in self.entries:
            tree.add(log._rich_render_())

        return tree


class InMemoryTaskSpan(InMemorySpan, TaskSpan):
    input: SerializeAsAny[PydanticSerializable]
    output: Optional[SerializeAsAny[PydanticSerializable]] = None

    def record_output(self, output: PydanticSerializable) -> None:
        self.output = output

    def _rich_render_(self) -> Tree:
        """Renders the trace via classes in the `rich` package"""
        tree = Tree(label=self.name)

        tree.add(_render_log_value(self.input, "Input"))

        for log in self.entries:
            tree.add(log._rich_render_())

        tree.add(_render_log_value(self.output, "Output"))

        return tree


# Required for sphinx, see also: https://docs.pydantic.dev/2.4/errors/usage_errors/#class-not-fully-defined
InMemorySpan.model_rebuild()
InMemoryTracer.model_rebuild()


class StartTask(BaseModel):
    """Represents the payload/entry of a log-line indicating that a `TaskSpan` was opened through `Tracer.task_span`.

    Attributes:
        uuid: A unique id for the opened `TaskSpan`.
        parent: The unique id of the parent element of opened `TaskSpan`.
            This could refer to either a surrounding `TaskSpan`, `Span` or the top-level `Tracer`.
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
    """Represents the payload/entry of a log-line indicating that a `Span` was opened through `Tracer.span`.

    Attributes:
        uuid: A unique id for the opened `Span`.
        parent: The unique id of the parent element of opened `TaskSpan`.
            This could refer to either a surrounding `TaskSpan`, `Span` or the top-level `Tracer`.
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
    """Represents a plain log-entry created through `Tracer.log`.

    Attributes:
        message: the message-parameter of `Tracer.log`
        value: the value-parameter of `Tracer.log`
        timestamp: the timestamp when `Tracer.log` was called.
        parent: The unique id of the parent element of the log.
            This could refer to either a surrounding `TaskSpan`, `Span` or the top-level `Tracer`.
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


class FileTracer(Tracer):
    """A `Tracer` that logs to a file.

    Each log-entry is represented by a JSON object. The information logged allows
    to reconstruct the hierarchical nature of the logs, i.e. all entries have a
    _pointer_ to its parent element in form of a parent attribute containing
    the uuid of the parent.

    Args:
        log_file_path: Denotes the file to log to.

    Attributes:
        uuid: a uuid for the tracer. If multiple :class:`FileTracer` instances log to the same file
            the child-elements for a tracer can be identified by referring to this id as parent.
    """

    def __init__(self, log_file_path: Path) -> None:
        self._log_file_path = log_file_path
        self.uuid = uuid4()

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


class FileSpan(Span, FileTracer):
    """A `Span` created by `FileTracer.span`."""

    end_timestamp: Optional[datetime] = None

    def __init__(self, log_file_path: Path, name: str) -> None:
        super().__init__(log_file_path)

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

    def end(self, timestamp: Optional[datetime] = None) -> None:
        if not self.end_timestamp:
            self.end_timestamp = timestamp or datetime.utcnow()
            self._log_entry(EndSpan(uuid=self.uuid, end=self.end_timestamp))


class FileTaskSpan(TaskSpan, FileSpan):
    """A `TaskSpan` created by `FileTracer.task_span`."""

    output: Optional[PydanticSerializable] = None

    def __init__(
        self, log_file_path: Path, task_name: str, input: PydanticSerializable
    ) -> None:
        super().__init__(log_file_path, task_name)

    def record_output(self, output: PydanticSerializable) -> None:
        self.output = output

    def end(self, timestamp: Optional[datetime] = None) -> None:
        if not self.end_timestamp:
            self.end_timestamp = timestamp or datetime.utcnow()
            self._log_entry(
                EndTask(uuid=self.uuid, end=self.end_timestamp, output=self.output)
            )
