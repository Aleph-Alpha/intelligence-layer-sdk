from typing import Optional
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
    """Input for an `InstructionTask`.

    Attributes:
        instruction: A textual instruction for the model.
            Could be a directive to answer a question or to translate something.
        input: The text-input for the instruction, e.g. a text to be translated.
        model: The name of the model that should handle the instruction.
        response_prefix: A string that is provided to the LLM as a prefix of the response.
            This can steer the model completion.
        maximum_response_tokens: The maximum number of tokens to be generated in the answer.
            For generating answers the default probably works fine in case of translations this
            tpically depends on the text to be translated.
    """

    instruction: str
    input: Optional[str]
    model: str
    response_prefix: str = ""
    maximum_response_tokens: int = 64


class InstructionOutput(BaseModel):
    """Output of an `InstructionTask`.

    Attributes:
        response: the model's generated response to the instruction.
        prompt_with_metadata: To handle the instruction a specific `PromptTemplate` is used
            the template defines two `PromptRange`s:
            - "instruction": covering the instruction text as provided in the `InstructionInput`.
            - "input": covering the input text as provided in the `InstructionInput`.
            These can for example be used for downstream `TextHighlight` tasks.
    """

    response: str
    prompt_with_metadata: PromptWithMetadata


class Instruction(Task[InstructionInput, InstructionOutput]):
    """Makes the model react to a given instruction and input text with a corresponding response.

    This can be used for different types of instructions a LLM can handle, like translations or
    answering questions based on an input text.

    Args:
        client: Aleph Alpha client instance for running model related API calls.

    Attributes:
        INSTRUCTION_PROMPT_TEMPLATE: The prompt-template used to build the actual `Prompt` sent
            to the inference API.

    Example:
        >>> client = Client(token="YOUR_AA_TOKEN")
        >>> task = Instruction(client)
        >>> input = InstructionInput(
        >>>     instruction="Translates the following to test to German.",
        >>>     input="An apple a day, keeps the doctor away."
        >>> )
        >>> logger = InMemoryLogger(name="Instruction")
        >>> output = task.run(input, logger)
        >>> print(output.response)
        >>> "Eine Apfel am Tag hält den Arzt fern."
    """

    INSTRUCTION_PROMPT_TEMPLATE = """### Instruction:
{% promptrange instruction %}{{instruction}}{% endpromptrange %}
{% if input %}
### Input:
{% promptrange input %}{{input}}{% endpromptrange %}
{% endif %}
### Response:{{response_prefix}}"""

    def __init__(self, client: Client) -> None:
        super().__init__()
        self._client = client
        self._completion = RawCompletion(client)

    def run(self, input: InstructionInput, logger: DebugLogger) -> InstructionOutput:
        prompt_with_metadata = PromptTemplate(
            self.INSTRUCTION_PROMPT_TEMPLATE
        ).to_prompt_with_metadata(
            input=input.input,
            instruction=input.instruction,
            response_prefix=input.response_prefix,
        )
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
