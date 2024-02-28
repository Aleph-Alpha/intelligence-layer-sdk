from datetime import datetime
from json import loads
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from intelligence_layer.core.tracer.in_memory_tracer import InMemoryTracer
from intelligence_layer.core.tracer.persistent_tracer import (
    PersistentSpan,
    PersistentTaskSpan,
    PersistentTracer,
)
from intelligence_layer.core.tracer.tracer import LogLine, PydanticSerializable


class FileTracer(PersistentTracer):
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
        super().__init__()
        self._log_file_path = log_file_path

    def _log_entry(self, id: str, entry: BaseModel) -> None:
        with self._log_file_path.open(mode="a", encoding="utf-8") as f:
            f.write(
                LogLine(
                    trace_id=id, entry_type=type(entry).__name__, entry=entry
                ).model_dump_json()
                + "\n"
            )

    def span(
        self,
        name: str,
        timestamp: Optional[datetime] = None,
        trace_id: Optional[str] = None,
    ) -> "FileSpan":
        span = FileSpan(self._log_file_path, trace_id=self.ensure_id(trace_id))
        self._log_span(span, name, timestamp)
        return span

    def task_span(
        self,
        task_name: str,
        input: PydanticSerializable,
        timestamp: Optional[datetime] = None,
        trace_id: Optional[str] = None,
    ) -> "FileTaskSpan":
        task = FileTaskSpan(
            self._log_file_path,
            trace_id=self.ensure_id(trace_id),
        )
        self._log_task(task, task_name, input, timestamp)
        return task

    def trace(self, trace_id: Optional[str] = None) -> InMemoryTracer:
        with self._log_file_path.open("r") as f:
            traces = (LogLine.model_validate(loads(line)) for line in f)
            filtered_traces = (
                (line for line in traces if line.trace_id == trace_id)
                if trace_id is not None
                else traces
            )
            return self._parse_log(filtered_traces)


class FileSpan(PersistentSpan, FileTracer):
    """A `Span` created by `FileTracer.span`."""

    def id(self) -> str:
        return self.trace_id

    def __init__(self, log_file_path: Path, trace_id: str) -> None:
        super().__init__(log_file_path)
        self.trace_id = trace_id


class FileTaskSpan(PersistentTaskSpan, FileSpan):
    """A `TaskSpan` created by `FileTracer.task_span`."""

    def __init__(
        self,
        log_file_path: Path,
        trace_id: str,
    ) -> None:
        super().__init__(log_file_path, trace_id)
