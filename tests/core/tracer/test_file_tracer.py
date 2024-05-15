from pathlib import Path
from unittest.mock import Mock

import pytest
from pytest import fixture

from intelligence_layer.core import CompositeTracer, FileTracer, InMemoryTracer, Task
from intelligence_layer.core.tracer.persistent_tracer import TracerLogEntryFailed


@fixture
def file_tracer(tmp_path: Path) -> FileTracer:
    return FileTracer(tmp_path / "log.log")


def test_file_tracer(file_tracer: FileTracer, test_task: Task[str, str]) -> None:
    input = "input"
    expected = InMemoryTracer()

    test_task.run(input, CompositeTracer([expected, file_tracer]))

    log_tree = file_tracer.trace()
    assert log_tree == expected


def test_file_tracer_retrieves_correct_trace(
    file_tracer: FileTracer, test_task: Task[str, str]
) -> None:
    input = "input"
    expected = InMemoryTracer()
    compositeTracer = CompositeTracer([expected, file_tracer])
    test_task.run(input, compositeTracer, "ID1")
    test_task.run(input, file_tracer, "ID2")
    log_tree = file_tracer.trace("ID1")
    assert log_tree == expected


def test_file_tracer_handles_tracer_log_entry_failed_exception(
    file_tracer: FileTracer,
) -> None:
    file_tracer._log_entry = Mock(  # type: ignore[method-assign]
        side_effect=[TracerLogEntryFailed("Hi I am an error", "21"), None]
    )

    try:
        file_tracer.task_span(
            task_name="mock_task_name", input="42", timestamp=None, trace_id="21"
        )
    except Exception as exception:
        assert False, f"'Unexpected exception: {exception}"


def test_file_tracer_raises_non_log_entry_failed_exceptions(
    file_tracer: FileTracer,
) -> None:
    file_tracer._log_entry = Mock(side_effect=[Exception("Hi I am an error", "21")])  # type: ignore[method-assign]
    with pytest.raises(Exception):
        file_tracer.task_span(
            task_name="mock_task_name", input="42", timestamp=None, trace_id="21"
        )
