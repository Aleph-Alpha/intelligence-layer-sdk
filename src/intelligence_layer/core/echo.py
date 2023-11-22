from functools import lru_cache
from typing import NewType, Sequence

from aleph_alpha_client import CompletionRequest, Prompt, Tokens
from pydantic import BaseModel
from tokenizers import Encoding, Tokenizer  # type: ignore

from intelligence_layer.connectors.limited_concurrency_client import (
    AlephAlphaClientProtocol,
)
from intelligence_layer.core.complete import Complete, CompleteInput
from intelligence_layer.core.prompt_template import PromptTemplate
from intelligence_layer.core.task import Task, Token
from intelligence_layer.core.tracer import TaskSpan

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
        model: A valid Aleph Alpha model name.
    """

    prompt: Prompt
    expected_completion: str
    model: str


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
        client: Aleph Alpha client instance for running model related API calls.

    Example:
        >>> client = LimitedConcurrencyClient.from_token(token="AA_TOKEN")
        >>> task = EchoTask(client)
        >>> input = EchoTaskInput(
                prompt="This is a ",
                expected_completion="happy text",
                model="luminous-base",
            )
        >>> tracer = InMemoryTracer()
        >>> output = task.run(input, tracer)
        >>> print(output.tokens_with_log_probs[0].prob)
        0.6
    """

    PROMPT_TEMPLATE: PromptTemplate = PromptTemplate(
        "{{prompt}}{{expected_completion}}"
    )

    def __init__(self, client: AlephAlphaClientProtocol) -> None:
        super().__init__()
        self._client = client
        self._completion = Complete(client=client)

    def do_run(self, input: EchoInput, task_span: TaskSpan) -> EchoOutput:
        # We tokenize the prompt separately so we don't have an overlap in the tokens.
        # If we don't do this, the end of the prompt and expected completion can be merged into unexpected tokens.
        expected_completion_tokens = self._tokenize(
            input.expected_completion, input.model
        )
        prompt = self.PROMPT_TEMPLATE.to_prompt(
            prompt=self.PROMPT_TEMPLATE.embed_prompt(input.prompt),
            expected_completion=self.PROMPT_TEMPLATE.placeholder(
                Tokens.from_token_ids(
                    [token.token_id for token in expected_completion_tokens]
                )
            ),
        )
        completion_input = CompleteInput(
            request=self._completion_request(prompt=prompt),
            model=input.model,
        )
        output = self._completion.run(completion_input, task_span)
        assert output.response.completions[0].log_probs
        log_prob_dicts = output.response.completions[0].log_probs[
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

    def _completion_request(
        self,
        prompt: Prompt,
    ) -> CompletionRequest:
        return CompletionRequest(
            prompt=prompt,
            maximum_tokens=0,
            log_probs=0,
            tokens=True,
            echo=True,
        )

    def _tokenize(self, text: str, model: str) -> Sequence[Token]:
        # Turns the expected output into list of token ids. Important so that we know how many tokens
        # the label is and can retrieve the last N log probs for the label
        tokenizer = self.tokenizer(model)
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

    @lru_cache
    def tokenizer(self, model: str) -> Tokenizer:
        return self._client.tokenizer(model)
