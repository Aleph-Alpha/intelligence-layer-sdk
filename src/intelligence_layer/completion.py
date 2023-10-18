from aleph_alpha_client import Client, CompletionRequest, CompletionResponse
from pydantic import BaseModel
from intelligence_layer.task import DebugLogger, Task


class RawCompletionInput(BaseModel):
    """The input for a `RawCompletion` task.

    Attributes:
        request: aleph-alpha-client's `CompletionRequest`. This gives fine grained control
            over all completion parameters that are supported by Aleph Alpha's inference API.
        model: the name of the model that actually performs the completion.
    """

    request: CompletionRequest
    model: str


class RawCompletionOutput(BaseModel):
    """The output for a `RawCompletion` task.

    Attributes:
        response: aleph-alpha-client's `CompletionResponse` containing all details
            provided by Aleph Alpha's inference API.
    """

    response: CompletionResponse

    @property
    def completion(self) -> str:
        return self.response.completions[0].completion or ""


class RawCompletion(Task[RawCompletionInput, RawCompletionOutput]):
    """Performs a completion-request with access to all possible request parameters.

    Only use this task if non of the higher level tasks defined below works for
    you, as your completion request does not fit to the use-cases the higher level ones represent or
    you need to control request-parameters that are not exposed by them.

    Args:
        client: Aleph Alpha client instance for running model related API calls.
    """

    def __init__(self, client: Client) -> None:
        super().__init__()
        self.client = client

    def run(
        self, input: RawCompletionInput, logger: DebugLogger
    ) -> RawCompletionOutput:
        response = self.client.complete(
            input.request,
            model=input.model,
        )
        return RawCompletionOutput(response=response)
