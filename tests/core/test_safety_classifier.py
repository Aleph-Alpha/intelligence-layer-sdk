from typing import List

import pytest
from pytest import fixture

from intelligence_layer.core import (
    Language,
    LuminousControlModel,
    NoOpTracer,
    TextChunk,
)
from intelligence_layer.core.safety_classifier import (
    SafetyClassifier,
    SafetyClassifyInput,
    SafetyOutputFlag,
)


@fixture
def safety_classifier(
    luminous_control_model: LuminousControlModel,
) -> SafetyClassifier:
    return SafetyClassifier(model=None)


def test_safety_classifier_returns_no_flags_for_safe_input(
    safety_classifier: SafetyClassifier,
) -> None:
    text = "This is a safe text"
    input = SafetyClassifyInput(text=TextChunk(text))

    output = safety_classifier.run(input, NoOpTracer())
    assert len(output.flags) == 0


@pytest.mark.parametrize(
    "text, expected_labels",
    [
        ("I will kill you.", [SafetyOutputFlag.VIOLENCE]),
        ("I want to kill myself.", [SafetyOutputFlag.SELF_HARM]),
        ("You fucking asshole.", [SafetyOutputFlag.INSULT]),
        (
            "The holy lord is going to deliver us from evil.",
            [SafetyOutputFlag.RELIGIOUS],
        ),
    ],
)
def test_safety_classifier_returns_flags_for_unsafe_english_input(
    safety_classifier: SafetyClassifier,
    text: str,
    expected_labels: List[SafetyOutputFlag],
) -> None:
    input = SafetyClassifyInput(text=TextChunk(text))

    output = safety_classifier.run(input, NoOpTracer())
    assert output.flags == expected_labels


@pytest.mark.parametrize(
    "text, expected_labels",
    [
        ("Ich töte dich.", [SafetyOutputFlag.VIOLENCE]),
        ("Ich möchte mich umbringen.", [SafetyOutputFlag.SELF_HARM]),
        ("Du verdammtes Arschloch.", [SafetyOutputFlag.INSULT]),
        (
            "Gott ist gütig und wird uns von allem Bösen erlösen.",
            [SafetyOutputFlag.RELIGIOUS],
        ),
    ],
)
def test_safety_classifier_returns_flags_for_unsafe_german_input(
    safety_classifier: SafetyClassifier,
    text: str,
    expected_labels: List[SafetyOutputFlag],
) -> None:
    input = SafetyClassifyInput(text=TextChunk(text), language=Language("de"))

    output = safety_classifier.run(input, NoOpTracer())
    assert output.flags == expected_labels
