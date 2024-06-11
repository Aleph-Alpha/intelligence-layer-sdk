import time

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
    Llama3InstructModel,
    LuminousControlModel,
    NoOpTracer,
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


def test_tokenizer_caching_works(
    client: AlephAlphaClientProtocol,
) -> None:
    test = AlephAlphaModel("luminous-supreme-control")
    start = time.time()
    test.get_tokenizer()
    end = time.time()
    assert end - start > 0.1

    start = time.time()
    test.get_tokenizer()
    end = time.time()
    assert end - start < 0.001
    # still fast even with new instance
    test = AlephAlphaModel("luminous-supreme-control")
    start = time.time()
    test.get_tokenizer()
    end = time.time()
    assert end - start < 0.001
