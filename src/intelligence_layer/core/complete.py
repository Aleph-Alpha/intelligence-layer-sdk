from typing import Optional, Sequence

from aleph_alpha_client import Client, CompletionRequest, CompletionResponse, Prompt
from pydantic import BaseModel

from intelligence_layer.core.prompt_template import PromptTemplate, PromptWithMetadata
from intelligence_layer.core.task import Task
from intelligence_layer.core.tracer import Tracer


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


class Complete(Task[CompleteInput, CompleteOutput]):
    """Performs a completion request with access to all possible request parameters.

    Only use this task if non of the higher level tasks defined below works for
    you, as your completion request does not fit to the use-cases the higher level ones represent or
    you need to control request-parameters that are not exposed by them.

    Args:
        client: Aleph Alpha client instance for running model related API calls.
    """

    def __init__(self, client: Client) -> None:
        super().__init__()
        self._client = client

    def run(self, input: CompleteInput, logger: Tracer) -> CompleteOutput:
        response = self._client.complete(
            input.request,
            model=input.model,
        )
        return CompleteOutput(response=response)


class InstructInput(BaseModel):
    """The input for an `Instruct`.

    Attributes:
        instruction: A textual instruction for the model.
            Could be a directive to answer a question or to translate something.
        input: The text input for the instruction, e.g. a text to be translated.
        model: The name of the model that should handle the instruction.
            Certain models are optimized for handling such instruction tasks.
            Typically their name contains 'control', e.g. 'luminous-extended-control'.
        response_prefix: A string that is provided to the LLM as a prefix of the response.
            This can steer the model completion.
        maximum_response_tokens: The maximum number of tokens to be generated in the answer.
            The default corresponds to roughly one short paragraph.
    """

    instruction: str
    input: Optional[str]
    model: str
    response_prefix: str = ""
    maximum_response_tokens: int = 64


class PromptOutput(BaseModel):
    """The output of an `Instruct` or `FewShot` task.

    Attributes:
        response: The generated response to the instruction.
        prompt_with_metadata: To handle the instruction, a `PromptTemplate` is used.
            The template defines two `PromptRange` instances:
            - "instruction": covering the instruction text.
            - "input": covering the input text.
            These can for example be used for downstream `TextHighlight` tasks.
    """

    response: str
    prompt_with_metadata: PromptWithMetadata


class Instruct(Task[InstructInput, PromptOutput]):
    """Runs zero-shot instruction completions on a model.

    Can be used for various types of instructions a LLM could handle, like QA, summarization,
    translation and more.

    Args:
        client: Aleph Alpha client instance for running model related API calls.

    Attributes:
        INSTRUCTION_PROMPT_TEMPLATE: The prompt-template used to build the actual `Prompt` sent
            to the inference API.

    Example:
        >>> client = Client(os.getenv("AA_TOKEN"))
        >>> task = Instruction(client)
        >>> input = InstructionInput(
                instruction="Translate the following to text to German.",
                input="An apple a day keeps the doctor away."
            )
        >>> logger = InMemoryLogger(name="Instruction")
        >>> output = task.run(input, logger)
        >>> print(output.response)
        Ein Apfel am Tag hält den Arzt fern.
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
        self._completion = Complete(client)

    def run(self, input: InstructInput, logger: Tracer) -> PromptOutput:
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
        return PromptOutput(
            response=completion, prompt_with_metadata=prompt_with_metadata
        )

    def _complete(
        self, prompt: Prompt, maximum_tokens: int, model: str, logger: Tracer
    ) -> str:
        request = CompletionRequest(prompt, maximum_tokens=maximum_tokens)
        return self._completion.run(
            CompleteInput(request=request, model=model),
            logger,
        ).completion


class FewShotExample(BaseModel):
    input: str
    response: str


class FewShotConfig(BaseModel):
    """Config for a few-shot prompt without dynamic input.

    Attributes:
        instruction: A textual instruction for the model.
            Could be a directive to answer a question or to translate something.
        examples: A number of few shot examples to prime the model.
        input_prefix: The prefix for each `FewShotExample.input` as well as the final input.
        response_prefix: The prefix for each `FewShotExample.response` as well as the completion.
    """

    instruction: str
    examples: Sequence[FewShotExample]
    input_prefix: str
    response_prefix: str


class FewShotInput(BaseModel):
    """The input for a `FewShot` task.

    Attributes:
        few_shot_config: The configuration to be used for generating a response.
        input: The text input for the prompt, e.g. a text to be translated.
        model: The name of the model that should handle the prompt.
            Vanilla models work best with few-shot promptung.
            These include 'luminous-base', 'extended' & 'supreme'.
        maximum_response_tokens: The maximum number of tokens to be generated in the answer.
            The default corresponds to roughly one short paragraph.
    """

    few_shot_config: FewShotConfig
    input: str
    model: str
    maximum_response_tokens: int = 64


class FewShot(Task[FewShotInput, PromptOutput]):
    """Runs few-shot completions on a model.

    Vanilla models work best with a show-don't-tell approach. Few-shot prompts illustrate
    the output that is expected from the model.

    Args:
        client: Aleph Alpha client instance for running model related API calls.

    Attributes:
        FEW_SHOT_PROMPT_TEMPLATE: The prompt-template used to build the actual `Prompt` sent
            to the inference API.

    Example:
        >>> client = Client(os.getenv("AA_TOKEN"))
        >>> task = FewShot(client)
        >>> input = FewShotInput(
                input="What is the capital of Germany?",
                few_shot_config=FewShotConfig(
                    instruction="Answer each question.",
                    examples=[
                        FewShotExample(input="How high is Mount Everest?", response="8848 metres."),
                        FewShotExample(input="When was Caesar killed?", response="44 AD."),
                    ],
                    input_prefix="Question",
                    response_prefix="Answer",
                    model="luminous-base"
                )
            )
        >>> logger = InMemoryLogger(name="FewShot")
        >>> output = task.run(input, logger)
        >>> print(output.response)
        Berlin.
    """

    FEW_SHOT_PROMPT_TEMPLATE = """{% promptrange instruction %}{{instruction}}
{% for example in few_shot_examples %}###
{{input_prefix}}: {{ example.input }}
{{response_prefix}}: {{ example.response }}
{% endfor %}{% endpromptrange %}###
{{input_prefix}}: {% promptrange input %}{{input}}{% endpromptrange %}
{{response_prefix}}:"""

    def __init__(self, client: Client) -> None:
        super().__init__()
        self._client = client
        self._completion = Complete(client)

    def run(self, input: FewShotInput, logger: Tracer) -> PromptOutput:
        prompt_with_metadata = PromptTemplate(
            self.FEW_SHOT_PROMPT_TEMPLATE
        ).to_prompt_with_metadata(
            instruction=input.few_shot_config.instruction,
            input=input.input,
            few_shot_examples=[
                e.model_dump() for e in input.few_shot_config.examples
            ],  # liquid can't handle classes, thus serializing
            input_prefix=input.few_shot_config.input_prefix,
            response_prefix=input.few_shot_config.response_prefix,
        )
        completion = self._complete(
            prompt_with_metadata.prompt,
            input.maximum_response_tokens,
            input.model,
            logger,
        )
        return PromptOutput(
            response=completion, prompt_with_metadata=prompt_with_metadata
        )

    def _complete(
        self, prompt: Prompt, maximum_tokens: int, model: str, logger: Tracer
    ) -> str:
        request = CompletionRequest(
            prompt, maximum_tokens=maximum_tokens, stop_sequences=["###"]
        )
        return self._completion.run(
            CompleteInput(request=request, model=model),
            logger,
        ).completion
