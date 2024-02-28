from datetime import datetime
from typing import Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, SerializeAsAny
from rich.tree import Tree

from intelligence_layer.core.tracer.tracer import (
    EndSpan,
    EndTask,
    LogEntry,
    LogLine,
    PlainEntry,
    PydanticSerializable,
    Span,
    StartSpan,
    StartTask,
    TaskSpan,
    Tracer,
    _render_log_value,
    utc_now,
)


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
        self,
        name: str,
        timestamp: Optional[datetime] = None,
        trace_id: Optional[str] = None,
    ) -> "InMemorySpan":
        child = InMemorySpan(
            name=name,
            start_timestamp=timestamp or utc_now(),
            trace_id=self.ensure_id(trace_id),
        )
        self.entries.append(child)
        return child

    def task_span(
        self,
        task_name: str,
        input: PydanticSerializable,
        timestamp: Optional[datetime] = None,
        trace_id: Optional[str] = None,
    ) -> "InMemoryTaskSpan":
        child = InMemoryTaskSpan(
            name=task_name,
            input=input,
            start_timestamp=timestamp or utc_now(),
            trace_id=self.ensure_id(trace_id),
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


# Required for sphinx, see also: https://docs.pydantic.dev/2.4/errors/usage_errors/#class-not-fully-defined
InMemorySpan.model_rebuild()
InMemoryTracer.model_rebuild()
