from functools import wraps
from threading import Lock
from time import sleep
from typing import Callable

from intelligence_layer.core.task import MAX_CONCURRENCY, Task
from intelligence_layer.core.tracer import InMemoryTracer, NoOpTracer, TaskSpan, Tracer


class ConcurrencyCounter(Task[None, None]):
    max_concurrency_counter: int = 0
    concurrency_counter: int = 0

    def __init__(self) -> None:
        self.lock = Lock()

    def run(self, input: None, tracer: Tracer) -> None:
        with self.lock:
            self.concurrency_counter += 1
            self.max_concurrency_counter = max(
                self.max_concurrency_counter, self.concurrency_counter
            )

        sleep(0.01)
        with self.lock:
            self.concurrency_counter -= 1


def dummy_decorator(
    f: Callable[["BaseTask", None, Tracer], None]
) -> Callable[["BaseTask", None, Tracer], None]:
    @wraps(f)
    def wrap(
        self: "BaseTask",
        input: None,
        tracer: Tracer,
    ) -> None:
        return f(self, input, tracer)

    return wrap


class BaseTask(Task[None, None]):
    @dummy_decorator
    def run(self, input: None, tracer: Tracer) -> None:
        tracer.log("Plain", "Entry")


class SubTask(BaseTask):
    pass


def test_run_concurrently() -> None:
    task = ConcurrencyCounter()
    task.run_concurrently([None] * MAX_CONCURRENCY * 10, NoOpTracer())
    assert task.max_concurrency_counter == MAX_CONCURRENCY


def test_run_concurrently_limited() -> None:
    task = ConcurrencyCounter()
    limit_concurrency = MAX_CONCURRENCY // 2
    task.run_concurrently([None] * MAX_CONCURRENCY * 3, NoOpTracer(), limit_concurrency)
    assert task.max_concurrency_counter == limit_concurrency


def test_sub_tasks_do_not_introduce_multiple_task_spans() -> None:
    tracer = InMemoryTracer()

    SubTask().run(None, tracer)

    assert tracer.logs
    assert isinstance(tracer.logs[0], TaskSpan)
    assert tracer.logs[0].logs
    assert not isinstance(tracer.logs[0].logs[0], TaskSpan)
