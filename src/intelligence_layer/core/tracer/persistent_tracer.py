from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterable, Optional
from uuid import uuid4

from pydantic import BaseModel

from intelligence_layer.core.tracer.in_memory_tracer import InMemoryTracer, TreeBuilder
from intelligence_layer.core.tracer.tracer import (
    EndSpan,
    EndTask,
    LogLine,
    PlainEntry,
    PydanticSerializable,
    Span,
    StartSpan,
    StartTask,
    TaskSpan,
    Tracer,
    utc_now,
)


class PersistentTracer(Tracer, ABC):
    def __init__(self) -> None:
        self.uuid = uuid4()

    @abstractmethod
    def _log_entry(self, id: str, entry: BaseModel) -> None:
        pass

    @abstractmethod
    def trace(self, trace_id: str) -> InMemoryTracer:
        pass

    def _log_span(
        self, span: "PersistentSpan", name: str, timestamp: Optional[datetime] = None
    ) -> None:
        self._log_entry(
            span.id(),
            StartSpan(
                uuid=span.uuid,
                parent=self.uuid,
                name=name,
                start=timestamp or utc_now(),
                trace_id=span.id(),
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
                task_span.id(),
                StartTask(
                    uuid=task_span.uuid,
                    parent=self.uuid,
                    name=task_name,
                    start=timestamp or utc_now(),
                    input=input,
                    trace_id=task_span.id(),
                ),
            )
        except TracerLogEntryFailed as error:
            self._log_entry(
                task_span.id(),
                StartTask(
                    uuid=task_span.uuid,
                    parent=self.uuid,
                    name=task_name,
                    start=timestamp or utc_now(),
                    input=error.description,
                    trace_id=task_span.id(),
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


class PersistentSpan(Span, PersistentTracer):
    end_timestamp: Optional[datetime] = None

    def log(
        self,
        message: str,
        value: PydanticSerializable,
        timestamp: Optional[datetime] = None,
    ) -> None:
        try:
            self._log_entry(
                self.id(),
                PlainEntry(
                    message=message,
                    value=value,
                    timestamp=timestamp or utc_now(),
                    parent=self.uuid,
                    trace_id=self.id(),
                ),
            )
        except TracerLogEntryFailed as error:
            self._log_entry(
                self.id(),
                PlainEntry(
                    message="log entry failed",
                    value=error.description,
                    timestamp=timestamp or utc_now(),
                    parent=self.uuid,
                    trace_id=self.id(),
                ),
            )

    def end(self, timestamp: Optional[datetime] = None) -> None:
        if not self.end_timestamp:
            self.end_timestamp = timestamp or utc_now()
            self._log_entry(self.id(), EndSpan(uuid=self.uuid, end=self.end_timestamp))


class PersistentTaskSpan(TaskSpan, PersistentSpan):
    output: Optional[PydanticSerializable] = None

    def record_output(self, output: PydanticSerializable) -> None:
        self.output = output

    def end(self, timestamp: Optional[datetime] = None) -> None:
        if not self.end_timestamp:
            self.end_timestamp = timestamp or utc_now()
            self._log_entry(
                self.id(),
                EndTask(uuid=self.uuid, end=self.end_timestamp, output=self.output),
            )


class TracerLogEntryFailed(Exception):
    def __init__(self, error_message: str, id: str) -> None:
        super().__init__(
            f"Log entry with id {id} failed with error message {error_message}."
        )
        self.description = error_message
