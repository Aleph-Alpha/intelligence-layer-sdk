from threading import Lock
from time import sleep
from intelligence_layer.task import MAX_CONCURRENCY, DebugLogger, NoOpDebugLogger, Task


class ConcurrencyCounter(Task[None, None]):
    max_concurrency_counter: int = 0
    concurrency_counter: int = 0

    def __init__(self) -> None:
        self.lock = Lock()

    def run(self, input: None, logger: DebugLogger) -> None:
        with self.lock:
            self.concurrency_counter += 1
            self.max_concurrency_counter = max(
                self.max_concurrency_counter, self.concurrency_counter
            )

        sleep(0.01)
        with self.lock:
            self.concurrency_counter -= 1


def test_run_concurrently() -> None:
    task = ConcurrencyCounter()
    logger = NoOpDebugLogger()
    task.run_concurrently([(None, logger)] * MAX_CONCURRENCY * 2)
    assert task.max_concurrency_counter == MAX_CONCURRENCY


def test_run_concurrently_limited() -> None:
    task = ConcurrencyCounter()
    logger = NoOpDebugLogger()
    limit_concurrency = MAX_CONCURRENCY // 2
    task.run_concurrently([(None, logger)] * MAX_CONCURRENCY * 2, limit_concurrency)
    assert task.max_concurrency_counter == limit_concurrency
