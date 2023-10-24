from typing import Sequence
from aleph_alpha_client import Client
from pytest import fixture
import tokenizers  # type: ignore
from intelligence_layer.echo import EchoInput, EchoTask, Token, TokenWithProb
from intelligence_layer.task import NoOpDebugLogger


@fixture
def echo_task(client: Client) -> EchoTask:
    return EchoTask(client)


def tokenize_completion(
    expected_output: str, model: str, client: Client
) -> Sequence[Token]:
    tokenizer = client.tokenizer(model)
    assert tokenizer.pre_tokenizer
    tokenizer.pre_tokenizer.add_prefix_space = False
    tokens: tokenizers.Encoding = tokenizer.encode(expected_output)
    return [
        Token(token=token, token_id=token_id)
        for token, token_id in zip(tokens.tokens, tokens.ids)
    ]


def test_can_run_echo_task(echo_task: EchoTask) -> None:
    expected_completion = "good."
    input = EchoInput(
        prompt="The weather is",
        expected_completion=expected_completion,
        model="luminous-base",
    )
    tokens = tokenize_completion(expected_completion, input.model, echo_task._client)

    result = echo_task.run(input, logger=NoOpDebugLogger())

    assert len(tokens) == len(result.tokens_with_log_probs)
    assert all([isinstance(t, TokenWithProb) for t in result.tokens_with_log_probs])
    for token, result_token in zip(tokens, result.tokens_with_log_probs):
        assert token == result_token.token


def test_compare_tokens(echo_task: EchoTask) -> None:
    token1 = "ĠGastronomie"
    token2 = "Baby"

    prompt = token1[0 : len(token1) // 2]
    expected_completion = token1[-len(token1) // 2 :] + token2
    input = EchoInput(
        prompt=prompt,
        expected_completion=expected_completion,
        model="luminous-base",
    )

    tokens = tokenize_completion(expected_completion, input.model, echo_task._client)
    result = echo_task.run(input, logger=NoOpDebugLogger())

    assert len(tokens) == len(result.tokens_with_log_probs)

    assert all([isinstance(t, TokenWithProb) for t in result.tokens_with_log_probs])
    for token, result_token in zip(tokens, result.tokens_with_log_probs):
        assert token == result_token.token
