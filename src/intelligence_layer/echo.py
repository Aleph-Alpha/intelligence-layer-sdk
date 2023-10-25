from itertools import chain
from typing import NewType, Sequence

from aleph_alpha_client import Client, CompletionRequest, Prompt, TokenizationRequest
from pydantic import BaseModel
import tokenizers  # type: ignore

from intelligence_layer.completion import RawCompletion, RawCompletionInput
from intelligence_layer.task import DebugLogger, Task


LogProb = NewType("LogProb", float)
Probability = NewType("Probability", float)


class Token(BaseModel):
    """A token class containing the raw token and its id.

    Refers to a unit of text that a model reads, which can be as short as a single character or as long as a word.

    Attributes:
        token: The actual text represented by the token.
        token_id: ID assigned to the token by the specific tokenizer.
    """

    token: str
    token_id: int


class TokenWithProb(BaseModel):
    token: Token
    prob: Probability | LogProb


class EchoInput(BaseModel):
    """The input for an `EchoTask`.

    Attributes:
        prompt: The input text that serves as the starting point for the LLM.
        expected_completion: The desired completion based on the prompt.
            The likelihood of the tokens in this will be examined.
        model: A valid Aleph Alpha model name.
    """

    prompt: str
    expected_completion: str
    model: str


class EchoOutput(BaseModel):
    """The output of an `EchoTask`.

    Attributes:
        tokens_with_log_probs: Every token of the `expected_completion` of the
            `EchoInput` accompanied by its probability of having been generated
            in a completion scenario.
    """

    tokens_with_log_probs: Sequence[TokenWithProb]


class EchoTask(Task[EchoInput, EchoOutput]):
    """Task that returns probabilities of the completion based on the given model and prompt.

    Analyzes the likelihood of generating tokens in the expected completion based on
    a given prompt and model. Does not generate any tokens.

    Args:
        client: Aleph Alpha client instance for running model related API calls.
    """

    def __init__(self, client: Client) -> None:
        super().__init__()
        self._completion = RawCompletion(client=client)
        self._client = client

    def run(self, input: EchoInput, logger: DebugLogger) -> EchoOutput:
        # We tokenize the prompt and expected completion separately so we don't 
        # have an overlap in the tokens.
        # If we don't do this, the end of the prompt and expected completion can be merged into unexpected tokens.
        expected_completion_tokens = self._tokenize(
            input.expected_completion, input.model
        )
        prompt = self._to_aa_tokens_prompt(
            list(chain(self._tokenize(input.prompt, input.model), expected_completion_tokens))
        )
        completion_input = RawCompletionInput(
            request=self._completion_request(prompt=prompt),
            model=input.model,
        )
        output = self._completion.run(completion_input, logger)
        assert output.response.completions[0].log_probs
        log_prob_dicts = output.response.completions[0].log_probs[
            -len(expected_completion_tokens) :
        ]
        tokens_with_prob = []
        for token, log_prob in zip(expected_completion_tokens, log_prob_dicts, strict=True):
            assert token.token in log_prob
            tokens_with_prob.append(
                TokenWithProb(
                    token=token,
                    prob=LogProb(log_prob.get(token.token, 0.0) or 0.0),
                )
            )
        return EchoOutput(tokens_with_log_probs=tokens_with_prob)
    
    def _to_aa_tokens_prompt(self, tokens: Sequence[Token]) -> Prompt:
        return Prompt.from_tokens([token.token_id for token in tokens])

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

    def _tokenize(self, expected_output: str, model: str) -> Sequence[Token]:
        """Turns the expected output into list of token ids. Important so that we know how many tokens
        the label is and can retrieve the last N log probs for the label"""
        response = self._client.tokenize(
            request=TokenizationRequest(expected_output, tokens=True, token_ids=True),
            model=model,
        )
        tokenizer = self._client.tokenizer(model)
        assert tokenizer.pre_tokenizer
        tokenizer.pre_tokenizer.add_prefix_space = False
        tokens: tokenizers.Encoding = tokenizer.encode(expected_output)
        assert response.token_ids and response.tokens
        return [
            Token(token=token, token_id=token_id)
            for token, token_id in zip(tokens.tokens, tokens.ids)
        ]
