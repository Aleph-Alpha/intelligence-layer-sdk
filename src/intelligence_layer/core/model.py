from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Optional

from aleph_alpha_client import CompletionRequest, CompletionResponse
from pydantic import BaseModel
from tokenizers import Encoding, Tokenizer  # type: ignore

from intelligence_layer.connectors.limited_concurrency_client import (
    AlephAlphaClientProtocol,
)
from intelligence_layer.core.prompt_template import PromptTemplate, RichPrompt
from intelligence_layer.core.task import Task
from intelligence_layer.core.tracer import TaskSpan, Tracer


class CompleteInput(BaseModel):
    """The input for a `Complete` task.

    Attributes:
        request: Aleph Alpha `Client`'s `CompletionRequest`. This gives fine grained control
            over all completion parameters that are supported by Aleph Alpha's inference API.
    """

    request: CompletionRequest


class CompleteOutput(BaseModel):
    """The output of a `Complete` task.

    Attributes:
        response: Aleph Alpha `Client`'s `CompletionResponse` containing all details
            provided by Aleph Alpha's inference API.
    """

    response: CompletionResponse

    @property
    def completion(self) -> str:
        return self.response.completions[0].completion or ""

    @property
    def generated_tokens(self) -> int:
        return self.response.num_tokens_generated


class _Complete(Task[CompleteInput, CompleteOutput]):
    """Performs a completion request with access to all possible request parameters.

    Only use this task if non of the higher level tasks defined below works for
    you, as your completion request does not fit to the use-cases the higher level ones represent or
    you need to control request-parameters that are not exposed by them.

    Args:
        client: Aleph Alpha client instance for running model related API calls.
    """

    def __init__(self, client: AlephAlphaClientProtocol, model: str) -> None:
        super().__init__()
        self._client = client
        self._model = model

    def do_run(self, input: CompleteInput, task_span: TaskSpan) -> CompleteOutput:
        response = self._client.complete(
            input.request,
            model=self._model,
        )
        return CompleteOutput(response=response)


class AlephAlphaModel(ABC):
    def __init__(self, model: str, client: AlephAlphaClientProtocol) -> None:
        self._client = client
        self._complete = _Complete(self._client, model)
        self._model = model

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
    def _get_tokenizer(self) -> Tokenizer:
        return self._client.tokenizer(self._model)

    def tokenize(self, text: str) -> Encoding:
        return self._get_tokenizer().encode(text)


class ControlModel(AlephAlphaModel):
    INSTRUCTION_PROMPT_TEMPLATE = PromptTemplate(
        """{% promptrange instruction %}{{instruction}}{% endpromptrange %}
{% if input %}
{% promptrange input %}{{input}}{% endpromptrange %}
{% endif %}
### Response:{{response_prefix}}"""
    )

    def to_instruct_prompt(
        self,
        instruction: str,
        input: Optional[str] = None,
        response_prefix: Optional[str] = None,
    ) -> RichPrompt:
        return self.INSTRUCTION_PROMPT_TEMPLATE.to_rich_prompt(
            instruction=instruction, input=input, response_prefix=response_prefix
        )
