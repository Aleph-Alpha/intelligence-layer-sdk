from aleph_alpha_client import Client, Text
from pytest import fixture
from intelligence_layer.core.completion import Instruction, InstructionInput
from intelligence_layer.task import NoOpDebugLogger


@fixture
def instruction(client: Client) -> Instruction:
    return Instruction(client)


def test_instruction_without_input(
    instruction: Instruction, no_op_debug_logger: NoOpDebugLogger
) -> None:
    input = InstructionInput(
        instruction="What is the capital of Germany?", input=None, model="luminous-base"
    )
    output = instruction.run(input, no_op_debug_logger)

    assert "Berlin" in output.response
    prompt_text_item = output.prompt_with_metadata.prompt.items[0]
    assert isinstance(prompt_text_item, Text)
    assert "Input" not in prompt_text_item.text
