from typing import Sequence

from pytest import fixture

from intelligence_layer.core import LuminousControlModel, NoOpTracer, TextChunk
from intelligence_layer.examples import (
    ClassifyInput,
    LabelWithDefinition,
    PromptBasedClassifyWithDefinitions,
    SingleLabelClassifyOutput,
)


@fixture
def labels_with_definitions() -> Sequence[LabelWithDefinition]:
    return [
        LabelWithDefinition(
            name="Dinosaur", definition="Any text that is about dinosaurs."
        ),
        LabelWithDefinition(name="Plant", definition="Any text that is about plants."),
        LabelWithDefinition(
            name="Toy", definition="Everything that has something to do with toys."
        ),
    ]


@fixture
def prompt_based_classify_with_definitions(
    luminous_control_model: LuminousControlModel,
    labels_with_definitions: Sequence[LabelWithDefinition],
) -> PromptBasedClassifyWithDefinitions:
    return PromptBasedClassifyWithDefinitions(
        labels_with_definitions, luminous_control_model
    )


def test_prompt_based_classify_with_definitions_returns_score_for_all_labels(
    prompt_based_classify_with_definitions: PromptBasedClassifyWithDefinitions,
    labels_with_definitions: Sequence[LabelWithDefinition],
) -> None:
    classify_input = ClassifyInput(
        chunk=TextChunk("I love my cactus!"),
        labels=frozenset(label.name for label in labels_with_definitions),
    )

    classify_output = prompt_based_classify_with_definitions.run(
        classify_input, NoOpTracer()
    )

    # Output contains everything we expect
    assert isinstance(classify_output, SingleLabelClassifyOutput)
    assert classify_input.labels == set(r for r in classify_output.scores)
