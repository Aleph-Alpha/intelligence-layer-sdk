from typing import NewType, Sequence
from aleph_alpha_client import Client, CompletionRequest, Prompt, TokenizationRequest
from pydantic import BaseModel
import tokenizers
from intelligence_layer.completion import RawCompletion, RawCompletionInput
from intelligence_layer.prompt_template import PromptTemplate

from intelligence_layer.task import DebugLogger, Task

LogProb = NewType("LogProb", float)
Probability = NewType("Probability", float)


class Token(BaseModel):
    """A token class containing it's id and the raw token.

    This is used instead of the Aleph Alpha client Token class since this one is serializable,
    while the one from the client is not.
    """

    token: str
    token_id: int


class TokenWithProb(BaseModel):
    token: Token
    prob: Probability | LogProb


class EchoInput(BaseModel):
    prompt: str
    expected_completion: str
    model: str


class EchoOutput(BaseModel):
    tokens_with_log_probs: Sequence[TokenWithProb]


class EchoTask(Task[EchoInput, EchoOutput]):
    """Task that returns probabilities of the completion based on the given model and prompt.

    This task asks the model how likely the given completion is. It does not generate any tokens.

    Args:
        client: Aleph Alpha client instance for running model related API calls.

    Example:

    """

    def __init__(self, client: Client) -> None:
        super().__init__()
        self._completion = RawCompletion(client=client)
        self._client = client

    def run(self, input: EchoInput, logger: DebugLogger) -> EchoOutput:
        prompt = Prompt.from_text(input.prompt + input.expected_completion)
        completion_input = RawCompletionInput(
            request=self._completion_request(prompt=prompt),
            model=input.model,
        )

        output = self._completion.run(completion_input, logger)
        tokens = self._tokenize_expected_completion(
            input.expected_completion, input.model
        )
        log_prob_dicts = output.response.completions[0].log_probs[-len(tokens) :]
        tokens_with_prob = []
        for token, log_prob in zip(tokens, log_prob_dicts):
            assert log_prob is not None
            tokens_with_prob.append(
                TokenWithProb(
                    token=token,
                    prob=LogProb(list(log_prob.values())[0]),
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

    def _tokenize_expected_completion(
        self, expected_output: str, model: str
    ) -> Sequence[Token]:
        """Turns th expected output into list of token ids. Important so that we know how many tokens
        the label is and can retrieve the last N log probs for the label"""
        response = self._client.tokenize(
            request=TokenizationRequest(expected_output, tokens=True, token_ids=True),
            model=model,
        )
        assert response.token_ids and response.tokens
        return [
            Token(token=token, token_id=token_id)
            for token, token_id in zip(response.tokens, response.token_ids)
        ]
