from aleph_alpha_client import CompletionRequest, Text
from pytest import fixture

from intelligence_layer.connectors.limited_concurrency_client import (
    AlephAlphaClientProtocol,
)
from intelligence_layer.core.model import AlephAlphaModel, CompleteInput, ControlModel
from intelligence_layer.core.tracer import NoOpTracer


@fixture
def model(client: AlephAlphaClientProtocol) -> AlephAlphaModel:
    return ControlModel(client=client, model="luminous-base-control-20240215")


def test_model_without_input(model: AlephAlphaModel, no_op_tracer: NoOpTracer) -> None:
    prompt = model.to_instruct_prompt("What is the capital of Germany?")
    input = CompleteInput(request=CompletionRequest(prompt=prompt))
    output = model.complete(input, no_op_tracer)

    assert "Berlin" in output.completion
    prompt_text_item = prompt.items[0]
    assert isinstance(prompt_text_item, Text)
    assert "Input" not in prompt_text_item.text
