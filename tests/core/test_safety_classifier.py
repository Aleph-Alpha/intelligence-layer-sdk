from pytest import fixture

from intelligence_layer.core import LuminousControlModel, NoOpTracer, TextChunk
from intelligence_layer.core.safety_classifier import (
    SafetyClassifier,
    SafetyClassifyInput,
)


@fixture
def safety_classifier(
    luminous_control_model: LuminousControlModel,
) -> SafetyClassifier:
    return SafetyClassifier(model=luminous_control_model)


def test_safety_classifier_returns_no_flags_for_safe_input(
    safety_classifier: SafetyClassifier,
) -> None:
    text = "This is a safe text"
    input = SafetyClassifyInput(text=TextChunk(text))

    output = safety_classifier.run(input, NoOpTracer())
    assert len(output.flags) == 0


def test_safety_classifier_returns_flags_for_unsafe_input(
    safety_classifier: SafetyClassifier,
) -> None:
    text = "I will kill you."
    input = SafetyClassifyInput(text=TextChunk(text))

    output = safety_classifier.run(input, NoOpTracer())
    assert len(output.flags) > 0
