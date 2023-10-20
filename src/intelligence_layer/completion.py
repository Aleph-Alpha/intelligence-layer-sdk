from aleph_alpha_client import Client, CompletionRequest, CompletionResponse
from pydantic import BaseModel
from intelligence_layer.task import DebugLogger, Task


class CompletionInput(BaseModel):
    request: CompletionRequest
    model: str


class CompletionOutput(BaseModel):
    response: CompletionResponse

    @property
    def completion(self) -> str:
        return self.response.completions[0].completion or ""


class Completion(Task[CompletionInput, CompletionOutput]):
    def __init__(self, client: Client) -> None:
        super().__init__()
        self.client = client

    def run(self, input: CompletionInput, logger: DebugLogger) -> CompletionOutput:
        response = self.client.complete(
            input.request,
            model=input.model,
        )
        return CompletionOutput(response=response)
