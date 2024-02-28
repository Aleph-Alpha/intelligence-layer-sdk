from typing import NewType, Sequence

from aleph_alpha_client import Prompt, Tokens
from pydantic import BaseModel
from tokenizers import Encoding  # type: ignore

from intelligence_layer.core.model import CompleteInput, ControlModel
from intelligence_layer.core.prompt_template import PromptTemplate
from intelligence_layer.core.task import Task, Token
from intelligence_layer.core.tracer.tracer import TaskSpan

LogProb = NewType("LogProb", float)


class TokenWithLogProb(BaseModel):
    token: Token
    prob: LogProb


class EchoInput(BaseModel):
    """The input for an `EchoTask`.

    Attributes:
        prompt: The input text that serves as the starting point for the LLM.
        expected_completion: The desired completion based on the prompt.
            The likelihood of the tokens in this will be examined.
    """

    prompt: Prompt
    expected_completion: str


class EchoOutput(BaseModel):
    """The output of an `EchoTask`.

    Attributes:
        tokens_with_log_probs: Every token of the `expected_completion` of the
            `EchoInput` accompanied by its probability of having been generated
            in a completion scenario.
    """

    tokens_with_log_probs: Sequence[TokenWithLogProb]


class EchoTask(Task[EchoInput, EchoOutput]):
    """Task that returns probabilities of a completion given a prompt.

    Analyzes the likelihood of generating tokens in the expected completion based on
    a given prompt and model. Does not generate any tokens.

    Args:
        model: Control model to use in the task.

    Example:
        >>> from aleph_alpha_client import Prompt
        >>> from intelligence_layer.core import EchoTask,EchoInput, InMemoryTracer, LuminousControlModel

        >>> model = LuminousControlModel(name="luminous-base-control")
        >>> task = EchoTask(model)
        >>> input = EchoInput(
        ...     prompt=Prompt.from_text("This is a "),
        ...     expected_completion="happy text",
        ... )
        >>> tracer = InMemoryTracer()
        >>> output = task.run(input, tracer)
    """

    PROMPT_TEMPLATE_STR: str = "{{prompt}}{{expected_completion}}"

    def __init__(self, model: ControlModel) -> None:
        super().__init__()
        self._model = model

    def do_run(self, input: EchoInput, task_span: TaskSpan) -> EchoOutput:
        # We tokenize the prompt separately so we don't have an overlap in the tokens.
        # If we don't do this, the end of the prompt and expected completion can be merged into unexpected tokens.
        expected_completion_tokens = self._tokenize(input.expected_completion)
        prompt_template = PromptTemplate(self.PROMPT_TEMPLATE_STR)
        prompt = prompt_template.to_rich_prompt(
            prompt=prompt_template.embed_prompt(input.prompt),
            expected_completion=prompt_template.placeholder(
                Tokens.from_token_ids(
                    [token.token_id for token in expected_completion_tokens]
                )
            ),
        )
        output = self._model.complete(
            CompleteInput(
                prompt=prompt,
                maximum_tokens=0,
                log_probs=0,
                tokens=True,
                echo=True,
            ),
            task_span,
        )
        assert output.completions[0].log_probs
        log_prob_dicts = output.completions[0].log_probs[
            -len(expected_completion_tokens) :
        ]
        tokens_with_prob = []
        for token, log_prob in zip(
            expected_completion_tokens, log_prob_dicts, strict=True
        ):
            assert token.token in log_prob
            tokens_with_prob.append(
                TokenWithLogProb(
                    token=token,
                    prob=LogProb(log_prob.get(token.token, 0.0) or 0.0),
                )
            )
        return EchoOutput(tokens_with_log_probs=tokens_with_prob)

    def _tokenize(self, text: str) -> Sequence[Token]:
        # Turns the expected output into list of token ids. Important so that we know how many tokens
        # the label is and can retrieve the last N log probs for the label
        tokenizer = self._model.get_tokenizer()
        if tokenizer.pre_tokenizer:
            tokenizer.pre_tokenizer.add_prefix_space = False
        encoding: Encoding = tokenizer.encode(text)
        return [
            Token(
                token=tokenizer.decode([token_id], skip_special_tokens=False),
                token_id=token_id,
            )
            for token_id in encoding.ids
        ]
