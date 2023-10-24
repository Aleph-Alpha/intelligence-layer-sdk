from typing import Sequence
from aleph_alpha_client import Client, Prompt, TokenizationRequest
from pytest import fixture
import tokenizers

from intelligence_layer.echo import EchoInput, EchoTask, Token, TokenWithProb
from intelligence_layer.task import NoOpDebugLogger


@fixture
def echo_task(client: Client) -> EchoTask:
    return EchoTask(client)


def tokenize_completion(
    expected_output: str, model: str, client: Client
) -> Sequence[Token]:
    """Turns th expected output into list of token ids. Important so that we know how many tokens
    the label is and can retrieve the last N log probs for the label"""
    response = client.tokenize(
        request=TokenizationRequest(
            expected_output + "<|endoftext|>", tokens=True, token_ids=True
        ),
        model=model,
    )
    assert response.token_ids and response.tokens
    return [
        Token(token=token, token_id=token_id)
        for token, token_id in zip(response.tokens, response.token_ids)
    ]


def test_can_run_echo_task(echo_task: EchoTask) -> None:
    excpected_completion = "good."
    input = EchoInput(
        prompt="The weather is",
        expected_completion=excpected_completion,
        model="luminous-base",
    )
    tokens = tokenize_completion(excpected_completion, input.model, echo_task._client)

    result = echo_task.run(input, logger=NoOpDebugLogger())

    assert all([isinstance(t, TokenWithProb) for t in result.tokens_with_log_probs])
    for token, result_token in zip(tokens, result.tokens_with_log_probs):
        assert token == result_token.token
