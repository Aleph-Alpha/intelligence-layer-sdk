from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import sleep
from typing import cast

from aleph_alpha_client import CompletionRequest, CompletionResponse, Prompt
from pytest import fixture

from intelligence_layer.connectors.limited_concurrency_client import (
    AlephAlphaClientProtocol,
    LimitedConcurrencyClient,
)


class ConcurrencyCountingClient:
    max_concurrency_counter: int = 0
    concurrency_counter: int = 0

    def __init__(self) -> None:
        self.lock = Lock()

    def complete(self, request: CompletionRequest, model: str) -> CompletionResponse:
        with self.lock:
            self.concurrency_counter += 1
            self.max_concurrency_counter = max(
                self.max_concurrency_counter, self.concurrency_counter
            )
        sleep(0.01)
        with self.lock:
            self.concurrency_counter -= 1
        return CompletionResponse(
            model_version="model-version",
            completions=[],
            optimized_prompt=None,
            num_tokens_generated=0,
            num_tokens_prompt_total=0,
        )


TEST_MAX_CONCURRENCY = 3


@fixture
def concurrency_counting_client() -> ConcurrencyCountingClient:
    return ConcurrencyCountingClient()


@fixture
def limited_concurrency_client(
    concurrency_counting_client: ConcurrencyCountingClient,
) -> LimitedConcurrencyClient:
    return LimitedConcurrencyClient(
        cast(AlephAlphaClientProtocol, concurrency_counting_client),
        TEST_MAX_CONCURRENCY,
    )


def test_methods_concurrency_is_limited(
    limited_concurrency_client: LimitedConcurrencyClient,
    concurrency_counting_client: ConcurrencyCountingClient,
) -> None:
    with ThreadPoolExecutor(max_workers=TEST_MAX_CONCURRENCY * 10) as executor:
        executor.map(
            limited_concurrency_client.complete,
            [CompletionRequest(prompt=Prompt(""))] * TEST_MAX_CONCURRENCY * 10,
            ["model"] * TEST_MAX_CONCURRENCY * 10,
        )
    assert concurrency_counting_client.max_concurrency_counter == TEST_MAX_CONCURRENCY
