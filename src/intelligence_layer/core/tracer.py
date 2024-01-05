from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Generic, Mapping, Optional, Sequence, TypeVar, Union
from uuid import UUID, uuid4

from opentelemetry.context import attach, detach
from opentelemetry.trace import Span as OpenTSpan
from opentelemetry.trace import Tracer as OpenTTracer
from opentelemetry.trace import set_span_in_context
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
        | None
        | bool
        | BaseModel
        | UUID
        | Sequence["PydanticSerializable"]
        | set["PydanticSerializable"]
        | frozenset["PydanticSerializable"]
        | Mapping[str, "PydanticSerializable"]
    )
else:
    PydanticSerializable = TypeAliasType(
        "PydanticSerializable",
        int
        | float
        | str
        | None
        | bool
        | BaseModel
        | UUID
        | Sequence["PydanticSerializable"]
        | set["PydanticSerializable"]
        | frozenset["PydanticSerializable"]
        | Mapping[str, "PydanticSerializable"],
    )


def utc_now() -> datetime:
    """return datetime object with utc timezone.

    datetime.utcnow() returns a datetime object without timezone, so this function is preferred.
    """
    return datetime.now(timezone.utc)

def ensure_id(id: Optional[str] =None) -> str:
    """Returns a valid id for tracing.

    Args:
        id: current id to use if present.
    Returns:
        `id` if present, otherwise a new unique ID.
    
    """
    return id if id is not None else str(uuid4())

class Tracer(ABC):
    """Provides a consistent way to instrument a :class:`Task` with logging for each step of the
    workflow.

    A tracer needs to provide a way to collect an individual log, which should be serializable, and
    a way to generate nested spans, so that sub-tasks can emit logs that are grouped together.

    Implementations of how logs are collected and stored may differ. Refer to the individual
    documentation of each implementation to see how to use the resulting tracer.
    """

    @abstractmethod
    def span(
        self, name: str, timestamp: Optional[datetime] = None, id: Optional[str] = None
    ) -> "Span":
        """Generate a span from the current span or logging instance.

        Allows for grouping multiple logs and duration together as a single, logical step in the
        process.

        Each tracer implementation can decide on how it wants to represent this, but they should
        all capture the hierarchical nature of nested spans, as well as the idea of the duration of
        the span.

        Args:
            name: A descriptive name of what this span will contain logs about.
            timestamp: optional override of the starting timestamp. Otherwise should default to now.
            id: optional override of a trace id. Otherwise it creates a new default id.

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
        id: Optional[str] = None,
    ) -> "TaskSpan":
        """Generate a task-specific span from the current span or logging instance.

        Allows for grouping multiple logs together, as well as the task's specific input, output,
        and duration.

        Each tracer implementation can decide on how it wants to represent this, but they should
        all allow for representing logs of a span within the context of a parent span.

        Args:
            task_name: The name of the task that is being logged
            input: The input for the task that is being logged.
            timestamp: optional override of the starting timestamp. Otherwise should default to now.
            id: optional override of a trace id. Otherwise it creates a new default id.


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

    @abstractmethod
    def id(self) -> str:
        ...

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
        >>> import os
        >>> from aleph_alpha_client import Client
        >>> from intelligence_layer.core import InMemoryTracer, FileTracer, CompositeTracer, Chunk
        >>> from intelligence_layer.use_cases import PromptBasedClassify, ClassifyInput

        >>> tracer_1 = InMemoryTracer()
        >>> tracer_2 = InMemoryTracer()
        >>> tracer = CompositeTracer([tracer_1, tracer_2])
        >>> aa_client = Client(os.getenv("AA_TOKEN"))
        >>> task = PromptBasedClassify(aa_client)
        >>> response = task.run(ClassifyInput(chunk=Chunk("Cool"), labels=frozenset({"label", "other label"})), tracer)
    """

    def __init__(self, tracers: Sequence[TracerVar]) -> None:
        assert len(tracers) > 0
        self.tracers = tracers

    def span(
        self, name: str, timestamp: Optional[datetime] = None, id: Optional[str] = None
    ) -> "CompositeSpan[Span]":
        timestamp = timestamp or utc_now()
        trace_id = ensure_id(id)
        return CompositeSpan(
            [tracer.span(name, timestamp, trace_id) for tracer in self.tracers]
        )

    def task_span(
        self,
        task_name: str,
        input: PydanticSerializable,
        timestamp: Optional[datetime] = None,
        id: Optional[str] = None,
    ) -> "CompositeTaskSpan":
        timestamp = timestamp or utc_now()
        trace_id = ensure_id(id)
        return CompositeTaskSpan(
            [
                tracer.task_span(task_name, input, timestamp, trace_id)
                for tracer in self.tracers
            ]
        )


SpanVar = TypeVar("SpanVar", bound=Span)


class CompositeSpan(Generic[SpanVar], CompositeTracer[SpanVar], Span):
    """A :class:`Span` that allows for recording to multiple spans simultaneously.

    Each log-entry and span will be forwarded to all subspans.

    Args:
        tracers: spans that will be forwarded all subsequent log and span calls.
    """

    def id(self) -> str:
        return self.tracers[0].id()

    def span(
        self, name: str, timestamp: Optional[datetime] = None, id: Optional[str] = None
    ) -> "CompositeSpan[Span]":
        return super().span(name, timestamp, id if id is not None else self.id())

    def task_span(
        self,
        task_name: str,
        input: PydanticSerializable,
        timestamp: Optional[datetime] = None,
        id: Optional[str] = None,
    ) -> "CompositeTaskSpan":
        return super().task_span(
            task_name, input, timestamp, id if id is not None else self.id()
        )

    def log(
        self,
        message: str,
        value: PydanticSerializable,
        timestamp: Optional[datetime] = None,
    ) -> None:
        timestamp = timestamp or utc_now()
        for tracer in self.tracers:
            tracer.log(message, value, timestamp)

    def end(self, timestamp: Optional[datetime] = None) -> None:
        timestamp = timestamp or utc_now()
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

    def id(self) -> str:
        return "NoOp"

    def log(
        self,
        message: str,
        value: PydanticSerializable,
        timestamp: Optional[datetime] = None,
    ) -> None:
        pass

    def span(
        self, name: str, timestamp: Optional[datetime] = None, id: Optional[str] = None
    ) -> "NoOpTracer":
        return self

    def task_span(
        self,
        task_name: str,
        input: PydanticSerializable,
        timestamp: Optional[datetime] = None,
        id: Optional[str] = None,
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
        id: The ID of the trace to which this log entry belongs.
    """

    message: str
    value: SerializeAsAny[PydanticSerializable]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    trace_id: str

    def id(self) -> str:
        return self.trace_id

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

    entries: list[Union[LogEntry, "InMemoryTaskSpan", "InMemorySpan"]] = []

    def span(
        self, name: str, timestamp: Optional[datetime] = None, id: Optional[str] = None
    ) -> "InMemorySpan":
        child = InMemorySpan(
            name=name,
            start_timestamp=timestamp or utc_now(),
            trace_id=ensure_id(id),
        )
        self.entries.append(child)
        return child

    def task_span(
        self,
        task_name: str,
        input: PydanticSerializable,
        timestamp: Optional[datetime] = None,
        id: Optional[str] = None,
    ) -> "InMemoryTaskSpan":
        child = InMemoryTaskSpan(
            name=task_name,
            input=input,
            start_timestamp=timestamp or utc_now(),
            trace_id=ensure_id(id),
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
    trace_id: str

    def id(self) -> str:
        return self.trace_id

    def span(
        self, name: str, timestamp: Optional[datetime] = None, id: Optional[str] = None
    ) -> "InMemorySpan":
        return super().span(name, timestamp, id if id is not None else self.id())

    def task_span(
        self,
        task_name: str,
        input: PydanticSerializable,
        timestamp: Optional[datetime] = None,
        id: Optional[str] = None,
    ) -> "InMemoryTaskSpan":
        return super().task_span(
            task_name, input, timestamp, id if id is not None else self.id()
        )

    def log(
        self,
        message: str,
        value: PydanticSerializable,
        timestamp: Optional[datetime] = None,
    ) -> None:
        self.entries.append(
            LogEntry(
                message=message,
                value=value,
                timestamp=timestamp or utc_now(),
                trace_id=self.id(),
            )
        )

    def end(self, timestamp: Optional[datetime] = None) -> None:
        if not self.end_timestamp:
            self.end_timestamp = timestamp or utc_now()

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
        trace_id: The trace id of the opened `TaskSpan`.

    """

    uuid: UUID
    parent: UUID
    name: str
    start: datetime
    input: SerializeAsAny[PydanticSerializable]
    trace_id: str


class EndTask(BaseModel):
    """Represents the payload/entry of a log-line that indicates that a `TaskSpan` ended (i.e. the context-manager exited).

    Attributes:
        uuid: The uuid of the corresponding `StartTask`.
        end: the timestamp when this `Task` completed (i.e. `run` returned).
        output: the `Output` (i.e. return value of `run`) the `Task` returned.
    """

    uuid: UUID
    end: datetime
    output: SerializeAsAny[PydanticSerializable]


class StartSpan(BaseModel):
    """Represents the payload/entry of a log-line indicating that a `Span` was opened through `Tracer.span`.

    Attributes:
        uuid: A unique id for the opened `Span`.
        parent: The unique id of the parent element of opened `TaskSpan`.
            This could refer to either a surrounding `TaskSpan`, `Span` or the top-level `Tracer`.
        name: The name of the task.
        start: The timestamp when this `Span` was started.
        trace_id: The ID of the trace this span belongs to.
    """

    uuid: UUID
    parent: UUID
    name: str
    start: datetime
    trace_id: str


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
        trace_id: The ID of the trace this entry belongs to.
    """

    message: str
    value: SerializeAsAny[PydanticSerializable]
    timestamp: datetime
    parent: UUID
    trace_id: str


class LogLine(BaseModel):
    """Represents a a complete log-line.

    Attributes:
        entry_type: The type of the entry. This is the class-name of one of the classes
            representing a log-entry (e.g. "StartTask").
        entry: The actual entry.

    """

    entry_type: str
    entry: SerializeAsAny[PydanticSerializable]


class TreeBuilder(BaseModel):
    root: InMemoryTracer = InMemoryTracer()
    tracers: dict[UUID, InMemoryTracer] = Field(default_factory=dict)
    tasks: dict[UUID, InMemoryTaskSpan] = Field(default_factory=dict)
    spans: dict[UUID, InMemorySpan] = Field(default_factory=dict)

    def start_task(self, log_line: LogLine) -> None:
        start_task = StartTask.model_validate(log_line.entry)
        child = InMemoryTaskSpan(
            name=start_task.name,
            input=start_task.input,
            start_timestamp=start_task.start,
            trace_id=start_task.trace_id,
        )
        self.tracers[start_task.uuid] = child
        self.tasks[start_task.uuid] = child
        self.tracers.get(start_task.parent, self.root).entries.append(child)

    def end_task(self, log_line: LogLine) -> None:
        end_task = EndTask.model_validate(log_line.entry)
        task_span = self.tasks[end_task.uuid]
        task_span.end_timestamp = end_task.end
        task_span.record_output(end_task.output)

    def start_span(self, log_line: LogLine) -> None:
        start_span = StartSpan.model_validate(log_line.entry)
        child = InMemorySpan(
            name=start_span.name,
            start_timestamp=start_span.start,
            trace_id=start_span.trace_id,
        )
        self.tracers[start_span.uuid] = child
        self.spans[start_span.uuid] = child
        self.tracers.get(start_span.parent, self.root).entries.append(child)

    def end_span(self, log_line: LogLine) -> None:
        end_span = EndSpan.model_validate(log_line.entry)
        span = self.spans[end_span.uuid]
        span.end_timestamp = end_span.end

    def plain_entry(self, log_line: LogLine) -> None:
        plain_entry = PlainEntry.model_validate(log_line.entry)
        entry = LogEntry(
            message=plain_entry.message,
            value=plain_entry.value,
            timestamp=plain_entry.timestamp,
            trace_id=plain_entry.trace_id,
        )
        self.tracers[plain_entry.parent].entries.append(entry)


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

    def span(
        self, name: str, timestamp: Optional[datetime] = None, id: Optional[str] = None
    ) -> "FileSpan":
        span = FileSpan(
            self._log_file_path, name, trace_id=ensure_id(id)
        )
        self._log_entry(
            StartSpan(
                uuid=span.uuid,
                parent=self.uuid,
                name=name,
                start=timestamp or utc_now(),
                trace_id=span.id(),
            )
        )
        return span

    def task_span(
        self,
        task_name: str,
        input: PydanticSerializable,
        timestamp: Optional[datetime] = None,
        id: Optional[str] = None,
    ) -> "FileTaskSpan":
        task = FileTaskSpan(
            self._log_file_path,
            task_name,
            input,
            trace_id=ensure_id(id),
        )
        self._log_entry(
            StartTask(
                uuid=task.uuid,
                parent=self.uuid,
                name=task_name,
                start=timestamp or utc_now(),
                input=input,
                trace_id=task.id(),
            )
        )
        return task


class FileSpan(Span, FileTracer):
    """A `Span` created by `FileTracer.span`."""

    end_timestamp: Optional[datetime] = None

    def id(self) -> str:
        return self.trace_id

    def __init__(self, log_file_path: Path, name: str, trace_id: str) -> None:
        super().__init__(log_file_path)
        self.trace_id = trace_id

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
                timestamp=timestamp or utc_now(),
                parent=self.uuid,
                trace_id=self.id(),
            )
        )

    def end(self, timestamp: Optional[datetime] = None) -> None:
        if not self.end_timestamp:
            self.end_timestamp = timestamp or utc_now()
            self._log_entry(EndSpan(uuid=self.uuid, end=self.end_timestamp))


class FileTaskSpan(TaskSpan, FileSpan):
    """A `TaskSpan` created by `FileTracer.task_span`."""

    output: Optional[PydanticSerializable] = None

    def __init__(
        self,
        log_file_path: Path,
        task_name: str,
        input: PydanticSerializable,
        trace_id: str,
    ) -> None:
        super().__init__(log_file_path, task_name, trace_id)

    def record_output(self, output: PydanticSerializable) -> None:
        self.output = output

    def end(self, timestamp: Optional[datetime] = None) -> None:
        if not self.end_timestamp:
            self.end_timestamp = timestamp or utc_now()
            self._log_entry(
                EndTask(uuid=self.uuid, end=self.end_timestamp, output=self.output)
            )


def _serialize(s: SerializeAsAny[PydanticSerializable]) -> str:
    value = s if isinstance(s, BaseModel) else JsonSerializer(root=s)
    return value.model_dump_json()


def _open_telemetry_timestamp(t: datetime) -> int:
    # Open telemetry expects *nanoseconds* since epoch
    t_float = t.timestamp() * 1e9
    return int(t_float)


class OpenTelemetryTracer(Tracer):
    """A `Tracer` that uses open telemetry."""

    def __init__(self, tracer: OpenTTracer) -> None:
        self._tracer = tracer

    def span(
        self, name: str, timestamp: Optional[datetime] = None, id: Optional[str] = None
    ) -> "OpenTelemetrySpan":
        trace_id = ensure_id(id)
        tracer_span = self._tracer.start_span(
            name,
            attributes={"trace_id": trace_id},
            start_time=None if not timestamp else _open_telemetry_timestamp(timestamp),
        )
        token = attach(set_span_in_context(tracer_span))
        return OpenTelemetrySpan(tracer_span, self._tracer, token, trace_id)

    def task_span(
        self,
        task_name: str,
        input: PydanticSerializable,
        timestamp: Optional[datetime] = None,
        id: Optional[str] = None,
    ) -> "OpenTelemetryTaskSpan":
        trace_id = ensure_id(id)

        tracer_span = self._tracer.start_span(
            task_name,
            attributes={"input": _serialize(input), "trace_id": trace_id},
            start_time=None if not timestamp else _open_telemetry_timestamp(timestamp),
        )
        token = attach(set_span_in_context(tracer_span))
        return OpenTelemetryTaskSpan(tracer_span, self._tracer, token, trace_id)


class OpenTelemetrySpan(Span, OpenTelemetryTracer):
    """A `Span` created by `OpenTelemetryTracer.span`."""

    end_timestamp: Optional[datetime] = None

    def id(self) -> str:
        return self._trace_id

    def __init__(
        self, span: OpenTSpan, tracer: OpenTTracer, token: object, trace_id: str
    ) -> None:
        super().__init__(tracer)
        self.open_ts_span = span
        self._token = token
        self._trace_id = trace_id

    def span(
        self, name: str, timestamp: Optional[datetime] = None, id: Optional[str] = None
    ) -> "OpenTelemetrySpan":
        return super().span(name, timestamp, id if id is not None else self.id())

    def task_span(
        self,
        task_name: str,
        input: PydanticSerializable,
        timestamp: Optional[datetime] = None,
        id: Optional[str] = None,
    ) -> "OpenTelemetryTaskSpan":
        return super().task_span(
            task_name, input, timestamp, id if id is not None else self.id()
        )

    def log(
        self,
        message: str,
        value: PydanticSerializable,
        timestamp: Optional[datetime] = None,
    ) -> None:
        self.open_ts_span.add_event(
            message,
            {"value": _serialize(value), "trace_id": self.id()},
            None if not timestamp else _open_telemetry_timestamp(timestamp),
        )

    def end(self, timestamp: Optional[datetime] = None) -> None:
        detach(self._token)
        self.open_ts_span.end(
            _open_telemetry_timestamp(timestamp) if timestamp is not None else None
        )


class OpenTelemetryTaskSpan(TaskSpan, OpenTelemetrySpan):
    """A `TaskSpan` created by `OpenTelemetryTracer.task_span`."""

    output: Optional[PydanticSerializable] = None

    def __init__(
        self, span: OpenTSpan, tracer: OpenTTracer, token: object, trace_id: str
    ) -> None:
        super().__init__(span, tracer, token, trace_id)

    def record_output(self, output: PydanticSerializable) -> None:
        self.open_ts_span.set_attribute("output", _serialize(output))
