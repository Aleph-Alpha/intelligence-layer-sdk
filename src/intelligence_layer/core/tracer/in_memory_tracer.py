import os
from datetime import datetime
from typing import Optional, Sequence, Union
from uuid import UUID

import requests
import rich
from pydantic import BaseModel, Field, SerializeAsAny
from requests import HTTPError
from rich.tree import Tree

from intelligence_layer.core.tracer.tracer import (
    Context,
    EndSpan,
    EndTask,
    Event,
    ExportedSpan,
    ExportedSpanList,
    LogEntry,
    LogLine,
    PlainEntry,
    PydanticSerializable,
    Span,
    SpanAttributes,
    StartSpan,
    StartTask,
    TaskSpan,
    TaskSpanAttributes,
    Tracer,
    _render_log_value,
    utc_now,
)


class InMemoryTracer(Tracer):
    """Collects log entries in a nested structure, and keeps them in memory.

    If desired, the structure is serializable with Pydantic, so you can write out the JSON
    representation to a file, or return via an API, or something similar.

    Attributes:
        name: A descriptive name of what the tracer contains log entries about.
        entries: A sequential list of log entries and/or nested InMemoryTracers with their own
            log entries.
    """

    def __init__(self) -> None:
        self.entries: list[Union[LogEntry, "InMemoryTaskSpan", "InMemorySpan"]] = []

    def span(
        self,
        name: str,
        timestamp: Optional[datetime] = None,
    ) -> "InMemorySpan":
        child = InMemorySpan(
            name=name,
            start_timestamp=timestamp or utc_now(),
            context=self.context,
        )
        self.entries.append(child)
        return child

    def task_span(
        self,
        task_name: str,
        input: PydanticSerializable,
        timestamp: Optional[datetime] = None,
    ) -> "InMemoryTaskSpan":
        child = InMemoryTaskSpan(
            name=task_name,
            input=input,
            start_timestamp=timestamp or utc_now(),
            context=self.context,
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

        if not self.submit_to_trace_viewer():
            rich.print(self._rich_render_())

    def submit_to_trace_viewer(self) -> bool:
        """Submits the trace to the UI for visualization"""
        trace_viewer_url = os.getenv("TRACE_VIEWER_URL", "http://localhost:3000")
        trace_viewer_trace_upload = f"{trace_viewer_url}/trace"
        try:
            res = requests.post(
                trace_viewer_trace_upload,
                json=ExportedSpanList(self.export_for_viewing()).model_dump_json(),
            )
            if res.status_code != 200:
                raise HTTPError(res.status_code)
            rich.print(
                f"Open the [link={trace_viewer_url}]Trace Viewer[/link] to view the trace."
            )
            return True
        except requests.ConnectionError:
            print(
                f"Trace viewer not found under {trace_viewer_url}.\nConsider running it for a better viewing experience.\nIf it is, set `TRACE_VIEWER_URL` in the environment."
            )
            return False

    def export_for_viewing(self) -> Sequence[ExportedSpan]:
        exported_root_spans: list[ExportedSpan] = []
        for entry in self.entries:
            if isinstance(entry, LogEntry):
                raise Exception(
                    "Found a log outside of a span. Logs can only be part of a span."
                )
            else:
                exported_root_spans.extend(entry.export_for_viewing())
        return exported_root_spans


class InMemorySpan(InMemoryTracer, Span):
    def __init__(
        self,
        name: str,
        context: Optional[Context] = None,
        start_timestamp: Optional[datetime] = None,
    ) -> None:
        InMemoryTracer.__init__(self)
        Span.__init__(self, context=context)
        self.parent_id = None if context is None else context.span_id
        self.name = name
        self.start_timestamp = (
            start_timestamp if start_timestamp is not None else utc_now()
        )
        self.end_timestamp: datetime | None = None

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
                trace_id=self.context.span_id,
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

    def _span_attributes(self) -> SpanAttributes:
        return SpanAttributes()

    def export_for_viewing(self) -> Sequence[ExportedSpan]:
        logs: list[LogEntry] = []
        exported_spans: list[ExportedSpan] = []
        for entry in self.entries:
            if isinstance(entry, LogEntry):
                logs.append(entry)
            else:
                exported_spans.extend(entry.export_for_viewing())
        exported_spans.append(
            ExportedSpan(
                context=self.context,
                name=self.name,
                parent_id=self.parent_id,
                start_time=self.start_timestamp,
                end_time=self.end_timestamp,
                attributes=self._span_attributes(),
                events=[
                    Event(
                        name="log",
                        body=log.value,
                        message=log.message,
                        timestamp=log.timestamp,
                    )
                    for log in logs
                ],
                status=self.status_code,
            )
        )
        return exported_spans


class InMemoryTaskSpan(InMemorySpan, TaskSpan):
    def __init__(
        self,
        name: str,
        input: SerializeAsAny[PydanticSerializable],
        context: Optional[Context] = None,
        start_timestamp: Optional[datetime] = None,
    ) -> None:
        super().__init__(name=name, context=context, start_timestamp=start_timestamp)
        self.input = input
        self.output: SerializeAsAny[PydanticSerializable] | None = None

    def record_output(self, output: PydanticSerializable) -> None:
        self.output = output

    def _span_attributes(self) -> SpanAttributes:
        return TaskSpanAttributes(input=self.input, output=self.output)

    def _rich_render_(self) -> Tree:
        """Renders the trace via classes in the `rich` package"""
        tree = Tree(label=self.name)

        tree.add(_render_log_value(self.input, "Input"))

        for log in self.entries:
            tree.add(log._rich_render_())

        tree.add(_render_log_value(self.output, "Output"))

        return tree


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
                context=Context(
                trace_id=start_task.trace_id, span_id=str(start_task.parent)
            ) if start_task.trace_id != str(start_task.uuid) else None
        )
        # if root, also change the trace id
        if converted_span.context.trace_id == converted_span.context.span_id:
            converted_span.context.trace_id = str(start_task.uuid)
        converted_span.context.span_id = str(start_task.uuid)
        self.tracers.get(start_task.parent, self.root).entries.append(converted_span)
        self.tracers[start_task.uuid] = converted_span
        self.tasks[start_task.uuid] = converted_span

    def end_task(self, log_line: LogLine) -> None:
        end_task = EndTask.model_validate(log_line.entry)
        task_span = self.tasks[end_task.uuid]
        task_span.record_output(end_task.output)
        task_span.end(end_task.end)

    def start_span(self, log_line: LogLine) -> None:
        start_span = StartSpan.model_validate(log_line.entry)
        converted_span = InMemorySpan(
            name=start_span.name,
            start_timestamp=start_span.start,
            context=Context(
                trace_id=start_span.trace_id, span_id=str(start_span.parent)
            ) if start_span.trace_id != str(start_span.uuid) else None
        )
        # if root, also change the trace id
        if converted_span.context.trace_id == converted_span.context.span_id:
            converted_span.context.trace_id = str(start_span.uuid)
        converted_span.context.span_id = str(start_span.uuid)
                
        self.tracers.get(start_span.parent, self.root).entries.append(converted_span)
        self.tracers[start_span.uuid] = converted_span
        self.spans[start_span.uuid] = converted_span

    def end_span(self, log_line: LogLine) -> None:
        end_span = EndSpan.model_validate(log_line.entry)
        span = self.spans[end_span.uuid]
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
