import traceback
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from enum import Enum
from types import TracebackType
from typing import TYPE_CHECKING, Literal, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, RootModel, SerializeAsAny
from typing_extensions import Self, TypeAliasType

# IMPORTANT: Only use this for converting your data to JSON. Do NOT use it for reading a JSON and convert it to an object.
# This could lead to loss of information (e.g. dictionaries getting dropped). For reading use JsonSerializable instead.
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
    """Return datetime object with utc timezone.

    datetime.utcnow() returns a datetime object without timezone, so this function is preferred.
    """
    return datetime.now(timezone.utc)


class Event(BaseModel):
    name: str
    message: str
    body: SerializeAsAny[PydanticSerializable]
    timestamp: datetime = Field(default_factory=utc_now)


class SpanType(str, Enum):
    SPAN = "SPAN"
    TASK_SPAN = "TASK_SPAN"


class SpanAttributes(BaseModel):
    type: Literal[SpanType.SPAN] = SpanType.SPAN


class TaskSpanAttributes(BaseModel):
    type: Literal[SpanType.TASK_SPAN] = SpanType.TASK_SPAN
    input: SerializeAsAny[PydanticSerializable]
    output: SerializeAsAny[PydanticSerializable]


class SpanStatus(Enum):
    OK = "OK"
    ERROR = "ERROR"


class Context(BaseModel):
    trace_id: UUID
    span_id: UUID


class ExportedSpan(BaseModel):
    context: Context
    name: str | None
    parent_id: UUID | None
    start_time: datetime
    end_time: datetime
    attributes: Union[SpanAttributes, TaskSpanAttributes] = Field(discriminator="type")
    events: Sequence[Event]
    status: SpanStatus
    # we ignore the links concept


ExportedSpanList = RootModel[Sequence[ExportedSpan]]


class Tracer(ABC):
    """Provides a consistent way to instrument a :class:`Task` with logging for each step of the workflow.

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
    ) -> "Span":
        """Generate a span from the current span or logging instance.

        Allows for grouping multiple logs and duration together as a single, logical step in the
        process.

        Each tracer implementation can decide on how it wants to represent this, but they should
        all capture the hierarchical nature of nested spans, as well as the idea of the duration of
        the span.

        Args:
            name: A descriptive name of what this span will contain logs about.
            timestamp: Override of the starting timestamp. Defaults to call time.

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
            timestamp: Override of the starting timestamp. Defaults to call time.

        Returns:
            An instance of a TaskSpan.
        """
        ...

    @abstractmethod
    def export_for_viewing(self) -> Sequence[ExportedSpan]:
        """Converts the trace to a format that can be read by Pharia Studio.

        The format is inspired by the OpenTelemetry Format, but does not abide by it.
        Specifically, it cuts away unused concepts, such as links.

        Returns:
            A list of spans which includes the current span and all its child spans.
        """
        ...


class ErrorValue(BaseModel):
    error_type: str
    message: str
    stack_trace: str


class Span(Tracer, AbstractContextManager["Span"]):
    """Captures a logical step within the overall workflow.

    Logs and other spans can be nested underneath.

    Can also be used as a context manager to easily capture the start and end time, and keep the
    span only in scope while it is open.

    Attributes:
        context: The context of the current span. If the span is a root span, the trace id will be equal to its span id.
        status_code: Status of the span. Will be "OK" unless the span was interrupted by an exception.
    """

    context: Context

    def __init__(self, context: Optional[Context] = None):
        """Creates a span from the context of its parent.

        Initializes the spans `context` based on the parent context and its `status_code`.

        Args:
            context: Context of the parent. Defaults to None.
        """
        span_id = uuid4()
        trace_id = span_id if context is None else context.trace_id
        self.context = Context(trace_id=trace_id, span_id=span_id)
        self.status_code = SpanStatus.OK
        self._closed = False

    def __enter__(self) -> Self:
        if self._closed:
            raise ValueError("Spans cannot be opened once they have been closed.")
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
        """Marks the Span as closed and sets the end time.

        The Span should be regarded as complete, and no further logging should happen with it.

        Ending a closed span in undefined behavior.

        Args:
            timestamp: Optional override of the timestamp. Defaults to call time.
        """
        self._closed = True

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
        """Record :class:`Task` output.

        Since a Context Manager can't capture output in the `__exit__` method,
        output should be captured once it is generated.

        Args:
            output: The output of the task that is being logged.
        """
        ...


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

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        _traceback: Optional[TracebackType],
    ) -> None:
        pass


class JsonSerializer(RootModel[PydanticSerializable]):
    root: SerializeAsAny[PydanticSerializable]
