import os
from datetime import datetime
from typing import Optional, Sequence, Union
from uuid import UUID

import requests
import rich
from pydantic import BaseModel, Field, SerializeAsAny
from requests import HTTPError
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree

from intelligence_layer.core.tracer.tracer import (
    Context,
    Event,
    ExportedSpan,
    ExportedSpanList,
    JsonSerializer,
    PydanticSerializable,
    Span,
    SpanAttributes,
    TaskSpan,
    TaskSpanAttributes,
    Tracer,
    utc_now,
)


class InMemoryTracer(Tracer):
    """Collects log entries in a nested structure, and keeps them in memory.

    Attributes:
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
    """A span that keeps all important information in memory.

    Attributes:
        context: Ids that uniquely describe the span.
        parent_id: Id of the parent span. None if the span is a root span.
        name: The name of the span.
        start_timestamp: The start of the timestamp.
        end_timestamp: The end of the timestamp. None until the span is closed.
        status_code: The status of the context.
    """

    def __init__(
        self,
        name: str,
        context: Optional[Context] = None,
        start_timestamp: Optional[datetime] = None,
    ) -> None:
        """Initializes a span and sets all necessary attributes.

        Args:
            name: The name of the span.
            context: The parent context. Used to derive the span's context. Defaults to None.
            start_timestamp: Custom start time of the span. Defaults to None.
        """
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
        super().end(timestamp)

    def _rich_render_(self) -> Tree:
        """Renders the trace via classes in the `rich` package"""
        tree = Tree(label=self.name)

        for log in self.entries:
            tree.add(log._rich_render_())

        return tree

    def _span_attributes(self) -> SpanAttributes:
        return SpanAttributes()

    def export_for_viewing(self) -> Sequence[ExportedSpan]:
        if not self._closed:
            raise RuntimeError(
                "Span is not closed. A Span must be closed before it is exported for viewing."
            )
        assert self.end_timestamp is not None

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
    """A span of a task that keeps all important information in memory.

    Attributes:
        context: Ids that uniquely describe the span.
        parent_id: Id of the parent span. None if the span is a root span.
        name: The name of the span.
        start_timestamp: The start of the timestamp.
        end_timestamp: The end of the timestamp. None until the span is closed.
        status_code: The status of the context.
        input: The input of the task.
        output: The output of the task.
    """

    def __init__(
        self,
        name: str,
        input: SerializeAsAny[PydanticSerializable],
        context: Optional[Context] = None,
        start_timestamp: Optional[datetime] = None,
    ) -> None:
        """Initializes a task span and sets all necessary attributes.

        Args:
            name: The name of the span.
            input: The input of a task. Needs to be serializable.
            context: The parent context. Used to derive the span's context. Defaults to None.
            start_timestamp: Custom start time of the span. Defaults to None.
        """
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
    trace_id: UUID

    def _rich_render_(self) -> Panel:
        """Renders the trace via classes in the `rich` package"""
        return _render_log_value(self.value, self.message)

    def _ipython_display_(self) -> None:
        """Default rendering for Jupyter notebooks"""
        from rich import print

        print(self._rich_render_())


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
