from aleph_alpha_client import ExplanationRequest, ExplanationResponse
from pydantic import BaseModel

from intelligence_layer.connectors.limited_concurrency_client import (
    AlephAlphaClientProtocol,
)
from intelligence_layer.core.model import AlephAlphaModel
from intelligence_layer.core.task import Task
from intelligence_layer.core.tracer import TaskSpan


class ExplainInput(BaseModel):
    """The input for a `Explain` task.

    Attributes:
        request: Aleph Alpha `Client`'s `ExplanationRequest`. This gives fine grained control
            over all explanation parameters that are supported by Aleph Alpha's inference API.
        model: A valid Aleph Alpha model name.
    """

    request: ExplanationRequest


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

    def __init__(self, model: AlephAlphaModel) -> None:
        super().__init__()
        self._model = model

    def do_run(self, input: ExplainInput, task_span: TaskSpan) -> ExplainOutput:
        response = self._model._client.explain(input.request, self._model.name)
        return ExplainOutput(response=response)
