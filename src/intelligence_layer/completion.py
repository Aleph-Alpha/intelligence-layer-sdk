from aleph_alpha_client import Client, CompletionRequest, CompletionResponse
from pydantic import BaseModel
from intelligence_layer.task import DebugLog, Task


class CompletionInput(BaseModel):
    request: CompletionRequest
    model: str


class CompletionOutput(BaseModel):
    response: CompletionResponse
    debug_log: DebugLog


class Completion(Task[CompletionInput, CompletionOutput]):
    def __init__(self, client: Client) -> None:
        super().__init__()
        self.client = client

    def run(self, input: CompletionInput) -> CompletionOutput:
        debug_log = DebugLog()
        debug_log.add("Request", {"request": input.request, "model": input.model})
        response = self.client.complete(
            input.request,
            model=input.model,
        )
        debug_log.add("Response", response)
        return CompletionOutput(response=response, debug_log=debug_log)
