import json
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
    LogEntry,
    LogLine,
    PlainEntry,
    PydanticSerializable,
    Span,
    SpanAttributes,
    SpanStatus,
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
                trace_viewer_trace_upload, json=json.loads(self.model_dump_json())
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
    name: str
    start_timestamp: datetime = Field(default_factory=datetime.utcnow)
    end_timestamp: Optional[datetime] = None

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
                context=Context(trace_id=self.id(), span_id="?"),
                name=self.name,
                parent_id=self.parent_id,
                start_time=self.start_timestamp,
                end_time=self.end_timestamp,
                attributes=SpanAttributes(),
                events=[
                    Event(
                        name="log",
                        body=log.value,
                        message=log.message,
                        timestamp=log.timestamp,
                    )
                    for log in logs
                ],
                status=SpanStatus.OK,
            )
        )
        return exported_spans


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
