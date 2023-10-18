from aleph_alpha_client import Client, CompletionRequest, CompletionResponse, Prompt
from pydantic import BaseModel
from intelligence_layer.prompt_template import PromptTemplate, PromptWithMetadata
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


class InstructionInput(BaseModel):
    instruction: str
    input: str
    maximum_response_tokens: int
    model: str


class InstructionOutput(BaseModel):
    response: str
    prompt_with_metadata: PromptWithMetadata


class Instruction(Task[InstructionInput, InstructionOutput]):
    INSTRUCTION_PROMPT_TEMPLATE = """### Instruction:
{% promptrange instruction %}{{instruction}}{% endpromptrange %}

### Input:
{% promptrange input %}{{input}}{% endpromptrange %}

### Response:"""

    def __init__(self, client: Client) -> None:
        super().__init__()
        self._client = client
        self._completion = RawCompletion(client)

    def run(self, input: InstructionInput, logger: DebugLogger) -> InstructionOutput:
        prompt_with_metadata = PromptTemplate(
            self.INSTRUCTION_PROMPT_TEMPLATE
        ).to_prompt_with_metadata(input=input.input, instruction=input.instruction)
        completion = self._complete(
            prompt_with_metadata.prompt,
            input.maximum_response_tokens,
            input.model,
            logger,
        )
        return InstructionOutput(
            response=completion, prompt_with_metadata=prompt_with_metadata
        )

    def _complete(
        self, prompt: Prompt, maximum_tokens: int, model: str, logger: DebugLogger
    ) -> str:
        request = CompletionRequest(prompt, maximum_tokens=maximum_tokens)
        return self._completion.run(
            RawCompletionInput(request=request, model=model),
            logger.child_logger("Completion"),
        ).completion
