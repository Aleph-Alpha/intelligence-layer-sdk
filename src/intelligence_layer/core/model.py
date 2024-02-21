from abc import ABC, abstractmethod
from dataclasses import asdict
from functools import lru_cache
from typing import Literal, Optional

from aleph_alpha_client import CompletionRequest, CompletionResponse
from pydantic import BaseModel, ConfigDict
from tokenizers import Encoding, Tokenizer  # type: ignore

from intelligence_layer.connectors.limited_concurrency_client import (
    AlephAlphaClientProtocol,
    LimitedConcurrencyClient,
)
from intelligence_layer.core.prompt_template import PromptTemplate, RichPrompt
from intelligence_layer.core.task import Task
from intelligence_layer.core.tracer import TaskSpan, Tracer


class CompleteInput(BaseModel, CompletionRequest, frozen=True):
    """The input for a `Complete` task."""

    pass


class CompleteOutput(BaseModel, CompletionResponse, frozen=True):
    """The output of a `Complete` task."""

    # BaseModel protects namespace "model_" but this is a field in the CompletionResponse
    model_config = ConfigDict(protected_namespaces=())

    @staticmethod
    def from_completion_response(
        completion_response: CompletionResponse,
    ) -> "CompleteOutput":
        return CompleteOutput(**asdict(completion_response))

    @property
    def completion(self) -> str:
        return self.completions[0].completion or ""

    @property
    def generated_tokens(self) -> int:
        return self.num_tokens_generated


class _Complete(Task[CompleteInput, CompleteOutput]):
    """Performs a completion request with access to all possible request parameters.

    Only use this task for testing. Is wrapped by the AlephAlphaModel for sending
    completion requests to the API.

    Args:
        client: Aleph Alpha client instance for running model related API calls.
        model: The name of a valid model that can access an API using an implementation
            of the AlephAlphaClientProtocol.
    """

    def __init__(self, client: AlephAlphaClientProtocol, model: str) -> None:
        super().__init__()
        self._client = client
        self._model = model

    def do_run(self, input: CompleteInput, task_span: TaskSpan) -> CompleteOutput:
        task_span.log("Model", self._model)
        return CompleteOutput.from_completion_response(
            self._client.complete(
                request=input,
                model=self._model,
            )
        )


class AlephAlphaModel(ABC):
    """Abstract base class for the implementation of any model that uses the Aleph Alpha client.

    Any class of Aleph Alpha model is implemented on top of this base class. Exposes methods that
    are available to all models, such as `complete` and `tokenize`. It is the central place for
    all things that are physically interconnected with a model, such as its tokenizer or prompt
    format used during training.

    Args:
        name: The name of a valid model that can access an API using an implementation
            of the AlephAlphaClientProtocol.
        client: Aleph Alpha client instance for running model related API calls.
    """

    def __init__(
        self,
        name: str,
        client: AlephAlphaClientProtocol,
    ) -> None:
        self.name = name
        self._client = client
        self._complete = _Complete(self._client, name)

    def get_complete_task(self) -> Task[CompleteInput, CompleteOutput]:
        return self._complete

    def complete(self, input: CompleteInput, tracer: Tracer) -> CompleteOutput:
        return self._complete.run(input, tracer)

    @abstractmethod
    def to_instruct_prompt(
        self,
        instruction: str,
        input: Optional[str] = None,
        response_prefix: Optional[str] = None,
    ) -> RichPrompt:
        ...

    @lru_cache(maxsize=1)
    def get_tokenizer(self) -> Tokenizer:
        return self._client.tokenizer(self.name)

    def tokenize(self, text: str) -> Encoding:
        return self.get_tokenizer().encode(text)


@lru_cache(maxsize=1)
def limited_concurrency_client_from_token() -> LimitedConcurrencyClient:
    return LimitedConcurrencyClient.from_token()


class LuminousControlModel(AlephAlphaModel):
    """An Aleph Alpha control model of the second generation.

    Args:
        name: The name of a valid model second generation control model.
        client: Aleph Alpha client instance for running model related API calls.
    """

    INSTRUCTION_PROMPT_TEMPLATE = PromptTemplate(
        """{% promptrange instruction %}{{instruction}}{% endpromptrange %}
{% if input %}
{% promptrange input %}{{input}}{% endpromptrange %}
{% endif %}
### Response:{{response_prefix}}"""
    )

    def __init__(
        self,
        name: Literal[
            "luminous-base-control",
            "luminous-extended-control",
            "luminous-supreme-control",
            "luminous-base-control-20240215",
            "luminous-extended-control-20240215",
            "luminous-supreme-control-20240215",
        ],
        client: AlephAlphaClientProtocol = limited_concurrency_client_from_token(),
    ) -> None:
        super().__init__(name, client)

    def to_instruct_prompt(
        self,
        instruction: str,
        input: Optional[str] = None,
        response_prefix: Optional[str] = None,
    ) -> RichPrompt:
        return self.INSTRUCTION_PROMPT_TEMPLATE.to_rich_prompt(
            instruction=instruction, input=input, response_prefix=response_prefix
        )
