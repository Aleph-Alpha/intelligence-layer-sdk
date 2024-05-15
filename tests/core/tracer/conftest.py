from pathlib import Path

from pytest import fixture

from intelligence_layer.core import Task, TaskSpan
from intelligence_layer.core.tracer.file_tracer import FileTracer
from intelligence_layer.core.tracer.in_memory_tracer import InMemoryTracer


class TestSubTask(Task[None, None]):
    def do_run(self, input: None, task_span: TaskSpan) -> None:
        task_span.log("subtask", "value")


class TestTask(Task[str, str]):
    sub_task = TestSubTask()

    def do_run(self, input: str, task_span: TaskSpan) -> str:
        with task_span.span("span") as sub_span:
            sub_span.log("message", "a value")
            self.sub_task.run(None, sub_span)
        self.sub_task.run(None, task_span)
        return "output"


class TestException(Exception):
    pass


@fixture
def test_task() -> Task[str, str]:
    return TestTask()


@fixture
def file_tracer(tmp_path: Path) -> FileTracer:
    return FileTracer(tmp_path / "log.log")


@fixture
def in_memory_tracer() -> InMemoryTracer:
    return InMemoryTracer()
