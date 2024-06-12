from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, SerializeAsAny

from intelligence_layer.core.tracer.in_memory_tracer import (
    InMemorySpan,
    InMemoryTaskSpan,
    InMemoryTracer,
    LogEntry,
)
from intelligence_layer.core.tracer.tracer import (
    Context,
    ExportedSpan,
    PydanticSerializable,
    Span,
    SpanStatus,
    TaskSpan,
    Tracer,
    utc_now,
)


class LogLine(BaseModel):
    """Represents a complete log-line.

    Attributes:
        entry_type: The type of the entry. This is the class-name of one of the classes
            representing a log-entry (e.g. "StartTask").
        entry: The actual entry.

    """

    trace_id: UUID
    entry_type: str
    entry: SerializeAsAny[PydanticSerializable]


class PersistentTracer(Tracer, ABC):
    def __init__(self) -> None:
        self.current_id = uuid4()

    @abstractmethod
    def _log_entry(self, id: UUID, entry: BaseModel) -> None:
        pass

    @abstractmethod
    def traces(self) -> InMemoryTracer:
        """Returns all traces of the given tracer.

        Returns:
            An InMemoryTracer that contains all traces of the tracer.
        """
        pass

    def export_for_viewing(self) -> Sequence[ExportedSpan]:
        return self.traces().export_for_viewing()

    def _log_span(
        self, span: "PersistentSpan", name: str, timestamp: Optional[datetime] = None
    ) -> None:
        self._log_entry(
            span.context.trace_id,
            StartSpan(
                uuid=span.context.span_id,
                parent=self.context.span_id if self.context else span.context.trace_id,
                name=name,
                start=timestamp or utc_now(),
                trace_id=span.context.trace_id,
            ),
        )

    def _log_task(
        self,
        task_span: "PersistentTaskSpan",
        task_name: str,
        input: PydanticSerializable,
        timestamp: Optional[datetime] = None,
    ) -> None:
        try:
            self._log_entry(
                task_span.context.trace_id,
                StartTask(
                    uuid=task_span.context.span_id,
                    parent=self.context.span_id
                    if self.context
                    else task_span.context.trace_id,
                    name=task_name,
                    start=timestamp or utc_now(),
                    input=input,
                    trace_id=task_span.context.trace_id,
                ),
            )
        except TracerLogEntryFailed as error:
            self._log_entry(
                task_span.context.trace_id,
                StartTask(
                    uuid=task_span.context.span_id,
                    parent=self.context.span_id
                    if self.context
                    else task_span.context.trace_id,
                    name=task_name,
                    start=timestamp or utc_now(),
                    input=error.description,
                    trace_id=task_span.context.trace_id,
                ),
            )

    def _parse_log(self, log_entries: Iterable[LogLine]) -> InMemoryTracer:
        tree_builder = TreeBuilder()
        for log_line in log_entries:
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


class PersistentSpan(Span, PersistentTracer, ABC):
    end_timestamp: Optional[datetime] = None

    def log(
        self,
        message: str,
        value: PydanticSerializable,
        timestamp: Optional[datetime] = None,
    ) -> None:
        try:
            self._log_entry(
                self.context.trace_id,
                PlainEntry(
                    message=message,
                    value=value,
                    timestamp=timestamp or utc_now(),
                    parent=self.context.span_id,
                    trace_id=self.context.trace_id,
                ),
            )
        except TracerLogEntryFailed as error:
            self._log_entry(
                self.context.trace_id,
                PlainEntry(
                    message="log entry failed",
                    value=error.description,
                    timestamp=timestamp or utc_now(),
                    parent=self.context.span_id,
                    trace_id=self.context.trace_id,
                ),
            )

    def end(self, timestamp: Optional[datetime] = None) -> None:
        if not self.end_timestamp:
            self.end_timestamp = timestamp or utc_now()
            self._log_entry(
                self.context.trace_id,
                EndSpan(
                    uuid=self.context.span_id,
                    end=self.end_timestamp,
                    status_code=self.status_code,
                ),
            )


class PersistentTaskSpan(TaskSpan, PersistentSpan, ABC):
    output: Optional[PydanticSerializable] = None

    def record_output(self, output: PydanticSerializable) -> None:
        self.output = output

    def end(self, timestamp: Optional[datetime] = None) -> None:
        if not self.end_timestamp:
            self.end_timestamp = timestamp or utc_now()
            self._log_entry(
                self.context.trace_id,
                EndTask(
                    uuid=self.context.span_id,
                    end=self.end_timestamp,
                    output=self.output,
                    status_code=self.status_code,
                ),
            )


class TracerLogEntryFailed(Exception):
    def __init__(self, error_message: str, id: str) -> None:
        super().__init__(
            f"Log entry with id {id} failed with error message {error_message}."
        )
        self.description = error_message


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
    trace_id: UUID


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
    status_code: SpanStatus = SpanStatus.OK


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
    trace_id: UUID


class EndSpan(BaseModel):
    """Represents the payload/entry of a log-line that indicates that a `Span` ended.

    Attributes:
        uuid: The uuid of the corresponding `StartSpan`.
        end: the timestamp when this `Span` completed.
    """

    uuid: UUID
    end: datetime
    status_code: SpanStatus = SpanStatus.OK


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
    trace_id: UUID


class TreeBuilder:
    def __init__(self) -> None:
        self.root = InMemoryTracer()
        self.tracers: dict[UUID, InMemoryTracer] = dict()
        self.tasks: dict[UUID, InMemoryTaskSpan] = dict()
        self.spans: dict[UUID, InMemorySpan] = dict()

    def start_task(self, log_line: LogLine) -> None:
        start_task = StartTask.model_validate(log_line.entry)
        converted_span = InMemoryTaskSpan(
            name=start_task.name,
            input=start_task.input,
            start_timestamp=start_task.start,
            context=Context(trace_id=start_task.trace_id, span_id=start_task.parent)
            if start_task.trace_id != start_task.uuid
            else None,
        )
        # if root, also change the trace id
        if converted_span.context.trace_id == converted_span.context.span_id:
            converted_span.context.trace_id = start_task.uuid
        converted_span.context.span_id = start_task.uuid
        self.tracers.get(start_task.parent, self.root).entries.append(converted_span)
        self.tracers[start_task.uuid] = converted_span
        self.tasks[start_task.uuid] = converted_span

    def end_task(self, log_line: LogLine) -> None:
        end_task = EndTask.model_validate(log_line.entry)
        task_span = self.tasks[end_task.uuid]
        task_span.record_output(end_task.output)
        task_span.status_code = end_task.status_code
        task_span.end(end_task.end)

    def start_span(self, log_line: LogLine) -> None:
        start_span = StartSpan.model_validate(log_line.entry)
        converted_span = InMemorySpan(
            name=start_span.name,
            start_timestamp=start_span.start,
            context=Context(trace_id=start_span.trace_id, span_id=start_span.parent)
            if start_span.trace_id != start_span.uuid
            else None,
        )
        # if root, also change the trace id
        if converted_span.context.trace_id == converted_span.context.span_id:
            converted_span.context.trace_id = start_span.uuid
        converted_span.context.span_id = start_span.uuid

        self.tracers.get(start_span.parent, self.root).entries.append(converted_span)
        self.tracers[start_span.uuid] = converted_span
        self.spans[start_span.uuid] = converted_span

    def end_span(self, log_line: LogLine) -> None:
        end_span = EndSpan.model_validate(log_line.entry)
        span = self.spans[end_span.uuid]
        span.status_code = end_span.status_code
        span.end(end_span.end)

    def plain_entry(self, log_line: LogLine) -> None:
        plain_entry = PlainEntry.model_validate(log_line.entry)
        entry = LogEntry(
            message=plain_entry.message,
            value=plain_entry.value,
            timestamp=plain_entry.timestamp,
            trace_id=plain_entry.trace_id,
        )
        self.tracers[plain_entry.parent].entries.append(entry)
