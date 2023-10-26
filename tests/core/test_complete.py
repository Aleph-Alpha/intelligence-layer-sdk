from aleph_alpha_client import Client, Text
from pytest import fixture

from intelligence_layer.core.complete import Instruct, InstructInput
from intelligence_layer.core.logger import NoOpDebugLogger


@fixture
def instruct(client: Client) -> Instruct:
    return Instruct(client)


def test_instruct_without_input(
    instruct: Instruct, no_op_debug_logger: NoOpDebugLogger
) -> None:
    input = InstructInput(
        instruction="What is the capital of Germany?", input=None, model="luminous-base"
    )
    output = instruct.run(input, no_op_debug_logger)

    assert "Berlin" in output.response
    prompt_text_item = output.prompt_with_metadata.prompt.items[0]
    assert isinstance(prompt_text_item, Text)
    assert "Input" not in prompt_text_item.text
