from aleph_alpha_client import Prompt, Text
from pytest import fixture

from intelligence_layer.connectors.limited_concurrency_client import (
    AlephAlphaClientProtocol,
)
from intelligence_layer.core import (
    AlephAlphaModel,
    CompleteInput,
    ControlModel,
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
