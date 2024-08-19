from collections.abc import Sequence
from typing import NewType

from aleph_alpha_client import Prompt, Text
from pydantic import BaseModel

from intelligence_layer.core.model import AlephAlphaModel
from intelligence_layer.core.task import Task, Token
from intelligence_layer.core.tracer.tracer import TaskSpan

LogProb = NewType("LogProb", float)


class TokenWithLogProb(BaseModel):
    token: Token
    prob: LogProb


class EchoInput(BaseModel):
    """The input for an `Echo` task.

    Attributes:
        prompt: The input text that serves as the starting point for the LLM.
        expected_completion: The desired completion based on the prompt.
            The likelihood of the tokens in this will be examined.
    """

    prompt: Prompt
    expected_completion: str


class EchoOutput(BaseModel):
    """The output of an `Echo` task.

    Attributes:
        tokens_with_log_probs: Every token of the `expected_completion` of the
            `EchoInput` accompanied by its probability of having been generated
            in a completion scenario.
    """

    tokens_with_log_probs: Sequence[TokenWithLogProb]


class Echo(Task[EchoInput, EchoOutput]):
    """Task that returns probabilities of a completion given a prompt.

    Analyzes the likelihood of generating tokens in the expected completion based on
    a given prompt and model. Does not generate any tokens.

    Args:
        model: A model to use in the task.

    Example:
        >>> from aleph_alpha_client import Prompt
        >>> from intelligence_layer.core import Echo, EchoInput, InMemoryTracer, LuminousControlModel

        >>> model = LuminousControlModel(name="luminous-base-control")
        >>> task = Echo(model)
        >>> input = EchoInput(
        ...     prompt=Prompt.from_text("This is a "),
        ...     expected_completion="happy text",
        ... )
        >>> tracer = InMemoryTracer()
        >>> output = task.run(input, tracer)
    """

    PROMPT_TEMPLATE_STR: str = "{{prompt}}{{expected_completion}}"

    def __init__(self, model: AlephAlphaModel) -> None:
        super().__init__()
        self._model = model

    def do_run(self, input: EchoInput, task_span: TaskSpan) -> EchoOutput:
        if len(input.prompt.items) != 1:
            raise NotImplementedError(
                "`Echo` currently only supports prompts with one item."
            )

        if not isinstance(input.prompt.items[0], Text):
            raise NotImplementedError(
                "`Echo` currently only supports prompts that are of type `Text`."
            )

        echo_output = self._model.echo(
            input.prompt.items[0].text, input.expected_completion, task_span
        )

        tokens_with_prob = [
            TokenWithLogProb(
                token=token,
                prob=LogProb(log_prob or 0.0),
            )
            for token, log_prob in echo_output
        ]
        return EchoOutput(tokens_with_log_probs=tokens_with_prob)
