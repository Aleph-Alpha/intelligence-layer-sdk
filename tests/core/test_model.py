import random
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from aleph_alpha_client import Prompt, PromptGranularity, Text
from pytest import fixture

from intelligence_layer.connectors import AlephAlphaClientProtocol
from intelligence_layer.core import (
    AlephAlphaModel,
    CompleteInput,
    ControlModel,
    ExplainInput,
    Llama2InstructModel,
    Llama3ChatModel,
    Llama3InstructModel,
    LuminousControlModel,
    Message,
    NoOpTracer,
)
from intelligence_layer.core.model import (
    _cached_context_size,
    _cached_tokenizer,
)


@fixture
def model(client: AlephAlphaClientProtocol) -> ControlModel:
    return LuminousControlModel(client=client, name="luminous-base-control")


@fixture
def base_model(client: AlephAlphaClientProtocol) -> AlephAlphaModel:
    return AlephAlphaModel(client=client, name="luminous-base")


def test_model_without_input(model: ControlModel, no_op_tracer: NoOpTracer) -> None:
    prompt = model.to_instruct_prompt("What is the capital of Germany?")
    input = CompleteInput(prompt=prompt)
    assert isinstance(input.model_dump(), dict)
    output = model.complete(input, no_op_tracer)
    assert isinstance(output.model_dump(), dict)

    assert "Berlin" in output.completion
    prompt_text_item = prompt.items[0]
    assert isinstance(prompt_text_item, Text)
    assert "Input" not in prompt_text_item.text


def test_aa_model(base_model: AlephAlphaModel, no_op_tracer: NoOpTracer) -> None:
    prompt = Prompt.from_text("The capital of Germany is")
    input = CompleteInput(prompt=prompt)

    output = base_model.complete(input, no_op_tracer)
    assert "Berlin" in output.completion


def test_explain(model: ControlModel, no_op_tracer: NoOpTracer) -> None:
    prompt = Prompt.from_text("What is the capital of Germany?")
    target = "Berlin."
    explain_input = ExplainInput(
        prompt=prompt, target=target, prompt_granularity=PromptGranularity.Word
    )
    output = model.explain(explain_input, no_op_tracer)
    assert output.explanations[0].items[0].scores[5].score > 1


def test_llama_2_model_works(no_op_tracer: NoOpTracer) -> None:
    llama_2_model = Llama2InstructModel()

    prompt = llama_2_model.to_instruct_prompt(
        "Who likes pizza?",
        "Marc and Jessica had pizza together. However, Marc hated it. He only agreed to the date because Jessica likes pizza so much.",
    )

    explain_input = CompleteInput(prompt=prompt)
    output = llama_2_model.complete(explain_input, no_op_tracer)
    assert "Jessica" in output.completion


def test_llama_3_model_works(no_op_tracer: NoOpTracer) -> None:
    llama_3_model = Llama3InstructModel()

    prompt = llama_3_model.to_instruct_prompt(
        "Who likes pizza?",
        "Marc and Jessica had pizza together. However, Marc hated it. He only agreed to the date because Jessica likes pizza so much.",
    )

    explain_input = CompleteInput(prompt=prompt)
    output = llama_3_model.complete(explain_input, no_op_tracer)
    assert "Jessica" in output.completion


def test_models_know_their_context_size(client: AlephAlphaClientProtocol) -> None:
    assert (
        LuminousControlModel(client=client, name="luminous-base-control").context_size
        == 2048
    )
    assert AlephAlphaModel(client=client, name="luminous-base").context_size == 2048
    assert (
        Llama2InstructModel(client=client, name="llama-2-7b-chat").context_size == 2048
    )
    assert (
        Llama3InstructModel(client=client, name="llama-3-8b-instruct").context_size
        == 8192
    )


def test_models_warn_about_non_recommended_models(
    client: AlephAlphaClientProtocol,
) -> None:
    with pytest.warns(UserWarning):
        assert LuminousControlModel(client=client, name="llama-2-7b-chat")  # type: ignore

    with pytest.warns(UserWarning):
        assert Llama2InstructModel(client=client, name="luminous-base")  # type: ignore

    with pytest.warns(UserWarning):
        assert Llama3InstructModel(client=client, name="llama-2-7b-chat")  # type: ignore

    with pytest.warns(UserWarning):
        assert AlephAlphaModel(client=client, name="No model")  # type: ignore


class DummyModelClient(AlephAlphaClientProtocol):
    # we use random here to simulate different objects to check caching behavior
    def tokenizer(self, model: str) -> float:
        return random.random()

    def models(self) -> Sequence[Mapping[str, Any]]:
        return [{"name": "model", "max_context_size": random.random()}]


def test_tokenizer_caching_works() -> None:
    client = DummyModelClient()  # type: ignore
    test = AlephAlphaModel("model", client=client)
    tokenizer = test.get_tokenizer()
    same_tokenizer = test.get_tokenizer()
    assert tokenizer is same_tokenizer

    another_model_instance = AlephAlphaModel("model", client=client)
    yet_same_tokenizer = another_model_instance.get_tokenizer()
    assert tokenizer is yet_same_tokenizer

    _cached_tokenizer.cache_clear()
    different_tokenizer = another_model_instance.get_tokenizer()
    assert tokenizer is not different_tokenizer


def test_context_size_caching_works() -> None:
    client = DummyModelClient()  # type: ignore
    test = AlephAlphaModel("model", client=client)
    context_size = test.context_size
    same_context_size = test.context_size
    assert context_size is same_context_size

    another_model_instance = AlephAlphaModel("model", client=client)
    yet_same_result = another_model_instance.context_size
    assert context_size is yet_same_result

    _cached_context_size.cache_clear()
    different_result = another_model_instance.context_size
    assert context_size is not different_result


def test_chat_model_can_produce_chat_prompt() -> None:
    client = DummyModelClient()  # type: ignore
    model = Llama3ChatModel("llama-3.1-8b-instruct", client)
    messages = [
        Message(role="system", content="You are a nice assistant."),
        Message(role="user", content="What's 2+2?"),
    ]
    response_prefix = "The answer is"

    prompt = model.to_chat_prompt(messages=messages, response_prefix=response_prefix)

    assert isinstance(prompt.items[0], Text)

    text_in_prompt = prompt.items[0].text

    assert (
        text_in_prompt
        == """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a nice assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What's 2+2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The answer is"""
    )
