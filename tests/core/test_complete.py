from aleph_alpha_client import Text
from pytest import fixture

from intelligence_layer.connectors.limited_concurrency_client import (
    AlephAlphaClientProtocol,
)
from intelligence_layer.core.complete import (
    FewShot,
    FewShotConfig,
    FewShotExample,
    FewShotInput,
    Instruct,
    InstructInput,
)
from intelligence_layer.core.tracer import NoOpTracer


@fixture
def instruct(client: AlephAlphaClientProtocol) -> Instruct:
    return Instruct(client)


@fixture
def few_shot(client: AlephAlphaClientProtocol) -> FewShot:
    return FewShot(client)


def test_instruct_without_input(instruct: Instruct, no_op_tracer: NoOpTracer) -> None:
    input = InstructInput(
        instruction="What is the capital of Germany?",
        input=None,
        model="luminous-base-control",
    )
    output = instruct.run(input, no_op_tracer)

    assert "Berlin" in output.completion
    prompt_text_item = output.prompt_with_metadata.prompt.items[0]
    assert isinstance(prompt_text_item, Text)
    assert "Input" not in prompt_text_item.text


def test_few_shot(few_shot: FewShot, no_op_tracer: NoOpTracer) -> None:
    input = FewShotInput(
        input="What is the capital of Germany?",
        few_shot_config=FewShotConfig(
            instruction="Answer each question.",
            examples=[
                FewShotExample(
                    input="How high is Mount Everest?", response="8848 metres."
                ),
                FewShotExample(input="When was Caesar killed?", response="44 AD."),
            ],
            input_prefix="Question",
            response_prefix="Answer",
        ),
        model="luminous-base",
    )
    output = few_shot.run(input, no_op_tracer)

    assert "Berlin" in output.completion
