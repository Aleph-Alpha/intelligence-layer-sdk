import time
from functools import lru_cache
from os import getenv
from threading import Semaphore
from typing import Any, Mapping, Optional, Protocol, Sequence

from aleph_alpha_client import (
    BatchSemanticEmbeddingRequest,
    BatchSemanticEmbeddingResponse,
    Client,
    CompletionRequest,
    CompletionResponse,
    DetokenizationRequest,
    DetokenizationResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    EvaluationRequest,
    EvaluationResponse,
    ExplanationRequest,
    ExplanationResponse,
    SemanticEmbeddingRequest,
    SemanticEmbeddingResponse,
    TokenizationRequest,
    TokenizationResponse,
)
from tokenizers import Tokenizer  # type: ignore


class AlephAlphaClientProtocol(Protocol):
    def complete(
        self,
        request: CompletionRequest,
        model: str,
    ) -> CompletionResponse:
        pass

    def get_version(self) -> str:
        pass

    def models(self) -> Sequence[Mapping[str, Any]]:
        pass

    def tokenize(
        self,
        request: TokenizationRequest,
        model: str,
    ) -> TokenizationResponse:
        pass

    def detokenize(
        self,
        request: DetokenizationRequest,
        model: str,
    ) -> DetokenizationResponse:
        pass

    def embed(
        self,
        request: EmbeddingRequest,
        model: str,
    ) -> EmbeddingResponse:
        pass

    def semantic_embed(
        self,
        request: SemanticEmbeddingRequest,
        model: str,
    ) -> SemanticEmbeddingResponse:
        pass

    def batch_semantic_embed(
        self,
        request: BatchSemanticEmbeddingRequest,
        model: Optional[str] = None,
    ) -> BatchSemanticEmbeddingResponse:
        pass

    def evaluate(
        self,
        request: EvaluationRequest,
        model: str,
    ) -> EvaluationResponse:
        pass

    def explain(
        self,
        request: ExplanationRequest,
        model: str,
    ) -> ExplanationResponse:
        pass

    def tokenizer(self, model: str) -> Tokenizer:
        pass


class LimitedConcurrencyClient:
    """An Aleph Alpha Client wrapper that limits the number of concurrent requests.

    This just delegates each call to the wrapped Aleph Alpha Client and ensures that
    never more then a given number of concurrent calls are executed against the API.

    Args:
        client: The wrapped `Client`.
        max_concurrency: the maximal number of requests that may run concurrently
            against the API.
        max_retry_time: the maximal time in seconds a complete is retried in case a `BusyError` is raised.

    """

    def __init__(
        self,
        client: AlephAlphaClientProtocol,
        max_concurrency: int = 20,
        max_retry_time: int = 600,
    ) -> None:
        self._client = client
        self._concurrency_limit_semaphore = Semaphore(max_concurrency)
        self._max_retry_time = max_retry_time

    @classmethod
    @lru_cache(maxsize=1)
    def from_env(
        cls, token: Optional[str] = None, host: Optional[str] = None
    ) -> "LimitedConcurrencyClient":
        """This is a helper method to construct your client with default settings from a token and host

        Args:
            token: An Aleph Alpha token to instantiate the client. If no token is provided,
                this method tries to fetch it from the environment under the name of "AA_TOKEN".
            host: The host that is used for requests. If no token is provided,
                this method tries to fetch it from the environment under the naem of "CLIENT_URL".
                If this is not present, it defaults to the Aleph Alpha Api.
                If you have an on premise setup, change this to your host URL.
        """
        if token is None:
            token = getenv("AA_TOKEN")
            assert token, "Define environment variable AA_TOKEN with a valid token for the Aleph Alpha API"
        if host is None:
            host = getenv("CLIENT_URL")
            if not host:
                host = "https://api.aleph-alpha.com"
                print(f"No CLIENT_URL specified in environment, using default: {host}.")

        return cls(Client(token, host=host))

    def complete(
        self,
        request: CompletionRequest,
        model: str,
    ) -> CompletionResponse:
        with self._concurrency_limit_semaphore:
            retries = 0
            start_time = time.time()
            latest_exception = None
            while time.time() - start_time < self._max_retry_time:
                try:
                    return self._client.complete(request, model)
                except Exception as e:
                    latest_exception = e
                    if e.args[0] == 503:
                        time.sleep(
                            min(
                                2**retries,
                                self._max_retry_time - (time.time() - start_time),
                            )
                        )
                        retries += 1
                        continue
                    else:
                        raise e
            assert latest_exception is not None
            raise latest_exception

    def get_version(self) -> str:
        with self._concurrency_limit_semaphore:
            return self._client.get_version()

    def models(self) -> Sequence[Mapping[str, Any]]:
        with self._concurrency_limit_semaphore:
            return self._client.models()

    def tokenize(
        self,
        request: TokenizationRequest,
        model: str,
    ) -> TokenizationResponse:
        with self._concurrency_limit_semaphore:
            return self._client.tokenize(request, model)

    def detokenize(
        self,
        request: DetokenizationRequest,
        model: str,
    ) -> DetokenizationResponse:
        with self._concurrency_limit_semaphore:
            return self._client.detokenize(request, model)

    def embed(
        self,
        request: EmbeddingRequest,
        model: str,
    ) -> EmbeddingResponse:
        with self._concurrency_limit_semaphore:
            return self._client.embed(request, model)

    def semantic_embed(
        self,
        request: SemanticEmbeddingRequest,
        model: str,
    ) -> SemanticEmbeddingResponse:
        with self._concurrency_limit_semaphore:
            return self._client.semantic_embed(request, model)

    def batch_semantic_embed(
        self,
        request: BatchSemanticEmbeddingRequest,
        model: Optional[str] = None,
    ) -> BatchSemanticEmbeddingResponse:
        with self._concurrency_limit_semaphore:
            return self._client.batch_semantic_embed(request, model)

    def evaluate(
        self,
        request: EvaluationRequest,
        model: str,
    ) -> EvaluationResponse:
        with self._concurrency_limit_semaphore:
            return self._client.evaluate(request, model)

    def explain(
        self,
        request: ExplanationRequest,
        model: str,
    ) -> ExplanationResponse:
        with self._concurrency_limit_semaphore:
            return self._client.explain(request, model)

    def tokenizer(self, model: str) -> Tokenizer:
        with self._concurrency_limit_semaphore:
            return self._client.tokenizer(model)
