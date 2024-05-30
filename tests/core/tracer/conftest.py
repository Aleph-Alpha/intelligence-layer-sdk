import time
from pathlib import Path

from pytest import fixture

from intelligence_layer.core import FileTracer, InMemoryTracer, Task, TaskSpan


class TracerTestSubTask(Task[None, None]):
    def do_run(self, input: None, task_span: TaskSpan) -> None:
        task_span.log("subtask", "value")


class TracerTestTask(Task[str, str]):
    sub_task = TracerTestSubTask()

    def do_run(self, input: str, task_span: TaskSpan) -> str:
        time.sleep(0.001)
        with task_span.span("span") as sub_span:
            time.sleep(0.001)
            sub_span.log("message", "a value")
            time.sleep(0.001)
            self.sub_task.run(None, sub_span)
            time.sleep(0.001)
        self.sub_task.run(None, task_span)
        time.sleep(0.001)
        return "output"


class SpecificTestException(Exception):
    pass


@fixture
def tracer_test_task() -> Task[str, str]:
    return TracerTestTask()


@fixture
def file_tracer(tmp_path: Path) -> FileTracer:
    return FileTracer(tmp_path / "log.log")


@fixture
def in_memory_tracer() -> InMemoryTracer:
    return InMemoryTracer()
