import traceback
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from enum import Enum
from types import TracebackType
from typing import TYPE_CHECKING, List, Mapping, Optional, Sequence
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, RootModel, SerializeAsAny
from rich.panel import Panel
from rich.syntax import Syntax
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


class Event(BaseModel):
    name: str
    message: str
    body: SerializeAsAny[PydanticSerializable]
    timestamp: datetime = Field(default_factory=utc_now)


class SpanType(Enum):
    SPAN = "SPAN"
    TASK_SPAN = "TASK_SPAN"


class SpanAttributes(BaseModel):
    type: SpanType = SpanType.SPAN


class TaskSpanAttributes(SpanAttributes):
    type: SpanType = SpanType.TASK_SPAN
    input: SerializeAsAny[PydanticSerializable]
    output: SerializeAsAny[PydanticSerializable]


class SpanStatus(Enum):
    OK = "OK"
    ERROR = "ERROR"


class Context(BaseModel):
    trace_id: str
    span_id: str


class ExportedSpan(BaseModel):
    context: Context
    name: str | None
    parent_id: str | None
    start_time: datetime
    end_time: datetime
    attributes: SpanAttributes
    events: Sequence[Event]
    status: SpanStatus
    # we ignore the links concept


class ExportedSpanList(RootModel):
    root: List[ExportedSpan]


class Tracer(ABC):
    """Provides a consistent way to instrument a :class:`Task` with logging for each step of the
    workflow.

    A tracer needs to provide a way to collect an individual log, which should be serializable, and
    a way to generate nested spans, so that sub-tasks can emit logs that are grouped together.

    Implementations of how logs are collected and stored may differ. Refer to the individual
    documentation of each implementation to see how to use the resulting tracer.
    """

    context: Context | None = None

    @abstractmethod
    def span(
        self,
        name: str,
        timestamp: Optional[datetime] = None,
    ) -> "Span":  # TODO
        """Generate a span from the current span or logging instance.

        Allows for grouping multiple logs and duration together as a single, logical step in the
        process.

        Each tracer implementation can decide on how it wants to represent this, but they should
        all capture the hierarchical nature of nested spans, as well as the idea of the duration of
        the span.

        Args:
            name: A descriptive name of what this span will contain logs about.
            timestamp: optional override of the starting timestamp. Otherwise should default to now.
            trace_id: optional override of a trace id. Otherwise it creates a new default id.

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
    ) -> "TaskSpan":  # TODO
        """Generate a task-specific span from the current span or logging instance.

        Allows for grouping multiple logs together, as well as the task's specific input, output,
        and duration.

        Each tracer implementation can decide on how it wants to represent this, but they should
        all allow for representing logs of a span within the context of a parent span.

        Args:
            task_name: The name of the task that is being logged
            input: The input for the task that is being logged.
            timestamp: optional override of the starting timestamp. Otherwise should default to now.
            trace_id: optional override of a trace id. Otherwise it creates a new default id.


        Returns:
            An instance of a TaskSpan.
        """
        ...

    def ensure_id(self, id: Optional[str]) -> str:
        """Returns a valid id for tracing.

        Args:
            id: current id to use if present.
        Returns:
            `id` if present, otherwise a new unique ID.
        """

        return id if id is not None else str(uuid4())

    @abstractmethod
    def export_for_viewing(self) -> Sequence[ExportedSpan]:
        """Converts the trace to a format that can be read by the trace viewer.

        The format is inspired by the OpenTelemetry Format, but does not abide by it,
        because it is too complex for our use-case.

        Returns:
            A list of spans which includes the current span and all its child spans.
        """
        ...


class ErrorValue(BaseModel):
    error_type: str
    message: str
    stack_trace: str


class Span(Tracer, AbstractContextManager["Span"]):
    """Captures a logical step within the overall workflow

    Logs and other spans can be nested underneath.

    Can also be used as a Context Manager to easily capture the start and end time, and keep the
    span only in scope while it is active.
    """

    def __init__(self, context: Optional[Context] = None):
        super().__init__()
        span_id = str(uuid4())
        if context is None:
            trace_id = span_id
        else:
            trace_id = context.trace_id
        self.context = Context(trace_id=trace_id, span_id=span_id)
        self.status_code = SpanStatus.OK
        self._closed = False

    def __enter__(self) -> Self:
        if self._closed:
            raise ValueError("Spans cannot be opened once they have been close.")
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

        Logging to closed spans is undefined behavior.

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

        Ending a closed span in undefined behavior.

        Args:
            timestamp: Optional override of the timestamp, otherwise should be set to now.
        """
        ...

    def ensure_id(self, id: str | None) -> str:
        """Returns a valid id for tracing.

        Args:
            id: current id to use if present.
        Returns:
            `id` if present, otherwise id of this `Span`
        """
        return id if id is not None else self.id()

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        _traceback: Optional[TracebackType],
    ) -> None:
        if exc_type is not None and exc_value is not None and _traceback is not None:
            error_value = ErrorValue(
                error_type=str(exc_type.__qualname__),
                message=str(exc_value),
                stack_trace=str(traceback.format_exc()),
            )
            self.log(error_value.message, error_value)
            self.status_code = SpanStatus.ERROR
        self.end()
        self._closed = True


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

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        _traceback: Optional[TracebackType],
    ) -> None:
        if exc_type is not None and exc_value is not None and _traceback is not None:
            error_value = ErrorValue(
                error_type=str(exc_type.__qualname__),
                message=str(exc_value),
                stack_trace=str(traceback.format_exc()),
            )
            self.record_output(error_value)
        self.end()


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
        self,
        name: str,
        timestamp: Optional[datetime] = None,
    ) -> "NoOpTracer":
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

    def export_for_viewing(self) -> Sequence[ExportedSpan]:
        return []


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

    def _rich_render_(self) -> Panel:
        """Renders the trace via classes in the `rich` package"""
        return _render_log_value(self.value, self.message)

    def _ipython_display_(self) -> None:
        """Default rendering for Jupyter notebooks"""
        from rich import print

        print(self._rich_render_())


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
    """Represents a complete log-line.

    Attributes:
        entry_type: The type of the entry. This is the class-name of one of the classes
            representing a log-entry (e.g. "StartTask").
        entry: The actual entry.

    """

    trace_id: str
    entry_type: str
    entry: SerializeAsAny[PydanticSerializable]


def _serialize(s: SerializeAsAny[PydanticSerializable]) -> str:
    value = s if isinstance(s, BaseModel) else JsonSerializer(root=s)
    return value.model_dump_json()
