from aleph_alpha_client import Client, CompletionRequest, CompletionResponse
from pydantic import BaseModel
from intelligence_layer.task import DebugLogger, Task


class CompletionInput(BaseModel):
    request: CompletionRequest
    model: str


class CompletionOutput(BaseModel):
    response: CompletionResponse

    def completion(self) -> str:
        return self.response.completions[0].completion or ""


class Completion(Task[CompletionInput, CompletionOutput]):
    def __init__(self, client: Client) -> None:
        super().__init__()
        self.client = client

    def run(self, input: CompletionInput, logger: DebugLogger) -> CompletionOutput:
        logger.log(
            "Request", {"request": input.request.to_json(), "model": input.model}
        )
        response = self.client.complete(
            input.request,
            model=input.model,
        )
        logger.log("Response", response.to_json())
        return CompletionOutput(response=response)
