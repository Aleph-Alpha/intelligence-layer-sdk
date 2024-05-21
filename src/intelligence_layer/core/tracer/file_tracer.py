from datetime import datetime
from json import loads
from pathlib import Path
from typing import Optional, Sequence

from pydantic import BaseModel

from intelligence_layer.core.tracer.in_memory_tracer import InMemoryTracer
from intelligence_layer.core.tracer.persistent_tracer import (
    PersistentSpan,
    PersistentTaskSpan,
    PersistentTracer,
)
from intelligence_layer.core.tracer.tracer import LogLine, PydanticSerializable
from intelligence_layer.core.tracer.tracer import (
    Context,
    Event,
    ExportedSpan,
    ExportedSpanList,
    LogEntry,
    PydanticSerializable,
    Span,
    SpanAttributes,
    TaskSpan,
    TaskSpanAttributes,
    Tracer,
    _render_log_value,
    utc_now,
)

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

    def __init__(self, log_file_path: Path | str, *args, **kwargs) -> None:
        super().__init__()
        self._log_file_path = Path(log_file_path)

    def _log_entry(self, id: str, entry: BaseModel) -> None:
        self._log_file_path.parent.mkdir(parents=True, exist_ok=True)
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
    ) -> "FileSpan":
        span = FileSpan(self._log_file_path, context=self.context)
        self._log_span(span, name, timestamp)
        return span

    def task_span(
        self,
        task_name: str,
        input: PydanticSerializable,
        timestamp: Optional[datetime] = None,
    ) -> "FileTaskSpan":
        task = FileTaskSpan(
            self._log_file_path,
            context=self.context,
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

    def __init__(self, log_file_path: Path, context: Optional[Context] = None) -> None:
        PersistentSpan.__init__(self, context=context)
        FileTracer.__init__(self, log_file_path=log_file_path)
    


class FileTaskSpan(PersistentTaskSpan, FileSpan):
    """A `TaskSpan` created by `FileTracer.task_span`."""
    pass
