from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from threading import Lock
from time import sleep
from typing import Callable

from intelligence_layer.core.task import MAX_CONCURRENCY, Task
from intelligence_layer.core.tracer import InMemoryTracer, NoOpTracer, TaskSpan


class ConcurrencyCounter(Task[None, None]):
    max_concurrency_counter: int = 0
    concurrency_counter: int = 0

    def __init__(self) -> None:
        self.lock = Lock()

    def do_run(self, input: None, task_span: TaskSpan) -> None:
        with self.lock:
            self.concurrency_counter += 1
            self.max_concurrency_counter = max(
                self.max_concurrency_counter, self.concurrency_counter
            )

        sleep(0.01)
        with self.lock:
            self.concurrency_counter -= 1


class DeadlockDetector(Task[None, None]):
    def __init__(self) -> None:
        super().__init__()
        self.inner_task = InnerTask()

    def do_run(self, input: None, task_span: TaskSpan) -> None:
        # wait a bit so all DeadlockDetector tasks run before the first InnerTask is submitted
        sleep(0.01)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self.inner_task.run_concurrently, [input], task_span
            )
            # wait a bit to ensure the future has finished
            # (even if the InnerTasks of all DeadlockDetector tasks are scheduled sequentially)
            for i in range(20):
                if future.done():
                    break
                sleep(0.1)
            if not future.done():
                executor.shutdown(wait=False)
                raise RuntimeError("Deadlock detected")


class InnerTask(Task[None, None]):
    def do_run(self, input: None, task_span: TaskSpan) -> None:
        pass


def dummy_decorator(
    f: Callable[["BaseTask", None, TaskSpan], None]
) -> Callable[["BaseTask", None, TaskSpan], None]:
    @wraps(f)
    def wrap(
        self: "BaseTask",
        input: None,
        task_span: TaskSpan,
    ) -> None:
        return f(self, input, task_span)

    return wrap


class BaseTask(Task[None, None]):
    @dummy_decorator
    def do_run(self, input: None, task_span: TaskSpan) -> None:
        task_span.log("Plain", "Entry")


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


def test_run_concurrently_does_not_deadlock_if_nested() -> None:
    task = DeadlockDetector()
    task.run_concurrently([None] * MAX_CONCURRENCY, NoOpTracer())


def test_sub_tasks_do_not_introduce_multiple_task_spans() -> None:
    tracer = InMemoryTracer()

    SubTask().run(None, tracer)

    assert tracer.entries
    assert isinstance(tracer.entries[0], TaskSpan)
    assert tracer.entries[0].entries
    assert not isinstance(tracer.entries[0].entries[0], TaskSpan)
