from aleph_alpha_client import Client, ExplanationRequest, ExplanationResponse
from pydantic import BaseModel

from intelligence_layer.core.task import Task
from intelligence_layer.core.tracer import Tracer


class ExplainInput(BaseModel):
    """The input for a `Explain` task.

    Attributes:
        request: Aleph Alpha `Client`'s `ExplanationRequest`. This gives fine grained control
            over all explanation parameters that are supported by Aleph Alpha's inference API.
        model: A valid Aleph Alpha model name.
    """

    request: ExplanationRequest
    model: str


class ExplainOutput(BaseModel):
    """The output of a `Explain` task.

    Attributes:
        response: Aleph Alpha `Client`'s `ExplanationResponse` containing all details
            provided by Aleph Alpha's inference API.
    """

    response: ExplanationResponse


class Explain(Task[ExplainInput, ExplainOutput]):
    """Performs an explanation request with access to all possible request parameters.

    Only use this task if non of the higher level tasks defined below works for
    you, for example if the `TextHighlight` task does not fit your use case.

    Args:
        client: Aleph Alpha client instance for running model related API calls.
    """

    def __init__(self, client: Client) -> None:
        super().__init__()
        self._client = client

    def run(self, input: ExplainInput, logger: Tracer) -> ExplainOutput:
        response = self._client.explain(input.request, input.model)
        return ExplainOutput(response=response)
