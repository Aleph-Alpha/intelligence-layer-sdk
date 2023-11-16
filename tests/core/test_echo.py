from typing import Sequence

import tokenizers  # type: ignore
from aleph_alpha_client import Prompt
from pytest import fixture

from intelligence_layer.connectors.limited_concurrency_client import (
    AlephAlphaClientProtocol,
)
from intelligence_layer.core.echo import EchoInput, EchoTask, TokenWithLogProb
from intelligence_layer.core.task import Token
from intelligence_layer.core.tracer import NoOpTracer


@fixture
def echo_task(client: AlephAlphaClientProtocol) -> EchoTask:
    return EchoTask(client)


def tokenize_completion(
    expected_output: str, model: str, client: AlephAlphaClientProtocol
) -> Sequence[Token]:
    tokenizer = client.tokenizer(model)
    assert tokenizer.pre_tokenizer
    tokenizer.pre_tokenizer.add_prefix_space = False
    encoding: tokenizers.Encoding = tokenizer.encode(expected_output)
    return [
        Token(
            token=tokenizer.decode([token_id], skip_special_tokens=False),
            token_id=token_id,
        )
        for token_id in encoding.ids
    ]


def test_can_run_echo_task(echo_task: EchoTask) -> None:
    expected_completion = "good."
    input = EchoInput(
        prompt=Prompt.from_text("The weather is"),
        expected_completion=expected_completion,
        model="luminous-base",
    )
    tokens = tokenize_completion(expected_completion, input.model, echo_task._client)

    result = echo_task.run(input, tracer=NoOpTracer())

    assert len(tokens) == len(result.tokens_with_log_probs)
    assert all([isinstance(t, TokenWithLogProb) for t in result.tokens_with_log_probs])
    for token, result_token in zip(tokens, result.tokens_with_log_probs):
        assert token == result_token.token


def test_echo_works_with_whitespaces_in_expected_completion(
    echo_task: EchoTask,
) -> None:
    expected_completion = " good."
    input = EchoInput(
        prompt=Prompt.from_text("The weather is"),
        expected_completion=expected_completion,
        model="luminous-base",
    )
    tokens = tokenize_completion(expected_completion, input.model, echo_task._client)

    result = echo_task.run(input, tracer=NoOpTracer())

    assert len(tokens) == len(result.tokens_with_log_probs)
    assert all([isinstance(t, TokenWithLogProb) for t in result.tokens_with_log_probs])
    for token, result_token in zip(tokens, result.tokens_with_log_probs):
        assert token == result_token.token


def test_overlapping_tokens_generate_correct_tokens(echo_task: EchoTask) -> None:
    """This test checks if the echo task correctly tokenizes the expected completion separately
    The two tokens when tokenized together will result in a combination of the end of the first token
    and the start of the second token. This is not the expected behaviour.
    """
    token1 = "ĠGastronomie"
    token2 = "Baby"

    prompt = Prompt.from_text(token1[0 : len(token1) // 2])
    expected_completion = token1[-len(token1) // 2 :] + token2
    input = EchoInput(
        prompt=prompt,
        expected_completion=expected_completion,
        model="luminous-base",
    )

    tokens = tokenize_completion(expected_completion, input.model, echo_task._client)
    result = echo_task.run(input, tracer=NoOpTracer())

    assert len(tokens) == len(result.tokens_with_log_probs)

    assert all([isinstance(t, TokenWithLogProb) for t in result.tokens_with_log_probs])
    for token, result_token in zip(tokens, result.tokens_with_log_probs):
        assert token == result_token.token
