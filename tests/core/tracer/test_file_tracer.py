from pathlib import Path
from unittest.mock import Mock

import pytest
from pytest import fixture

from intelligence_layer.core import (
    FileTracer,
    InMemoryTaskSpan,
    Task,
    TracerLogEntryFailed,
)
from tests.core.tracer.conftest import SpecificTestException


@fixture
def file_tracer(tmp_path: Path) -> FileTracer:
    return FileTracer(tmp_path / "log.log")


def test_file_tracer_retrieves_all_file_traces(
    file_tracer: FileTracer, tracer_test_task: Task[str, str]
) -> None:
    input = "input"

    tracer_test_task.run(input, file_tracer)
    tracer_test_task.run(input, file_tracer)
    traces = file_tracer.traces()
    assert len(traces.entries) == 2
    assert isinstance(traces.entries[0], InMemoryTaskSpan)
    assert isinstance(traces.entries[1], InMemoryTaskSpan)
    assert traces.entries[0].context.trace_id != traces.entries[1].context.trace_id


def test_file_tracer_handles_tracer_log_entry_failed_exception(
    file_tracer: FileTracer,
) -> None:
    file_tracer._log_entry = Mock(  # type: ignore[method-assign]
        side_effect=[TracerLogEntryFailed("Hi I am an error", "21"), None]
    )

    try:
        file_tracer.task_span(task_name="mock_task_name", input="42", timestamp=None)
    except Exception as exception:
        raise AssertionError(f"'Unexpected exception: {exception}") from None


def test_file_tracer_raises_non_log_entry_failed_exceptions(
    file_tracer: FileTracer,
) -> None:
    file_tracer._log_entry = Mock(  # type: ignore[method-assign]
        side_effect=[SpecificTestException("Hi I am an error", "21")]
    )
    with pytest.raises(SpecificTestException):
        file_tracer.task_span(task_name="mock_task_name", input="42", timestamp=None)


def test_file_tracer_is_backwards_compatible() -> None:
    current_file_location = Path(__file__)
    file_tracer = FileTracer(
        current_file_location.parent / "fixtures/old_file_trace_format.jsonl"
    )
    tracer = file_tracer.traces()

    assert len(tracer.entries) == 1
    task_span = tracer.entries[0]
    assert isinstance(task_span, InMemoryTaskSpan)
    assert task_span.input == "input"
    assert task_span.start_timestamp and task_span.end_timestamp
    assert task_span.start_timestamp < task_span.end_timestamp
