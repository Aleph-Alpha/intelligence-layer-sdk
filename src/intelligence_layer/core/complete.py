from typing import Optional, Sequence

from aleph_alpha_client import CompletionRequest, CompletionResponse, Prompt
from pydantic import BaseModel, Field

from intelligence_layer.connectors.limited_concurrency_client import (
    AlephAlphaClientProtocol,
)
from intelligence_layer.core.prompt_template import PromptTemplate, RichPrompt
from intelligence_layer.core.task import Task
from intelligence_layer.core.tracer import TaskSpan


class CompleteInput(BaseModel):
    """The input for a `Complete` task.

    Attributes:
        request: Aleph Alpha `Client`'s `CompletionRequest`. This gives fine grained control
            over all completion parameters that are supported by Aleph Alpha's inference API.
        model: A valid Aleph Alpha model name.
    """

    request: CompletionRequest
    model: str


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


class Complete(Task[CompleteInput, CompleteOutput]):
    """Performs a completion request with access to all possible request parameters.

    Only use this task if non of the higher level tasks defined below works for
    you, as your completion request does not fit to the use-cases the higher level ones represent or
    you need to control request-parameters that are not exposed by them.

    Args:
        client: Aleph Alpha client instance for running model related API calls.
    """

    def __init__(self, client: AlephAlphaClientProtocol) -> None:
        super().__init__()
        self._client = client

    def do_run(self, input: CompleteInput, task_span: TaskSpan) -> CompleteOutput:
        response = self._client.complete(
            input.request,
            model=input.model,
        )
        return CompleteOutput(response=response)
