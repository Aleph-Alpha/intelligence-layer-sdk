from threading import Semaphore
from typing import Any, Mapping, Optional, Protocol, Sequence

from aleph_alpha_client import (
    BatchSemanticEmbeddingRequest,
    BatchSemanticEmbeddingResponse,
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
        ...

    def get_version(self) -> str:
        ...

    def models(self) -> Sequence[Mapping[str, Any]]:
        ...

    def tokenize(
        self,
        request: TokenizationRequest,
        model: str,
    ) -> TokenizationResponse:
        ...

    def detokenize(
        self,
        request: DetokenizationRequest,
        model: str,
    ) -> DetokenizationResponse:
        ...

    def embed(
        self,
        request: EmbeddingRequest,
        model: str,
    ) -> EmbeddingResponse:
        ...

    def semantic_embed(
        self,
        request: SemanticEmbeddingRequest,
        model: str,
    ) -> SemanticEmbeddingResponse:
        ...

    def batch_semantic_embed(
        self,
        request: BatchSemanticEmbeddingRequest,
        model: Optional[str] = None,
    ) -> BatchSemanticEmbeddingResponse:
        ...

    def evaluate(
        self,
        request: EvaluationRequest,
        model: str,
    ) -> EvaluationResponse:
        ...

    def explain(
        self,
        request: ExplanationRequest,
        model: str,
    ) -> ExplanationResponse:
        ...

    def tokenizer(self, model: str) -> Tokenizer:
        ...


class LimitedConcurrencyClient:
    """A Aleph Alpha"""

    def __init__(self, client: AlephAlphaClientProtocol, max_concurrency: int) -> None:
        self._client = client
        self._concurrency_limit_semaphore = Semaphore(max_concurrency)

    def complete(
        self,
        request: CompletionRequest,
        model: str,
    ) -> CompletionResponse:
        with self._concurrency_limit_semaphore:
            return self._client.complete(request, model)

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
